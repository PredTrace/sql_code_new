import os
import sys
from verify_core_op import *
from ir import *
from z3_eval_plan_ir import *
from predicate_pushdown import predicate_pushdown_operator, obtain_subquery_filter, obtain_predicate_candidate

# pred is predicate on input
def obtain_predicate_candidate_pushup(op, preds):
    if isinstance(op, FilterInference):
        return And(preds[0], op.condition)
    elif isinstance(op, ProjectionInference):
        # input filter nation.key
        # output filter: n1.key
        #print("\t * * input filter : {}".format(expr_to_sql(op.input_filters[0])))
        
        ret = True
        equiv_pairs = {k.get_full_name():v for k,v in get_equivalent_pairs(preds[0])}
        for new_col,expr in op.col_func_pair: 
            if all([is_col_in_exist_columns(c, set([k for k,v in equiv_pairs.items()])) for c in get_columns_used(expr)]):
                newe = get_filter_replacing_field(expr, equiv_pairs)
                if str(newe) != str(expr):
                    newe = EqualTo(new_col, newe)
                    ret = newe if is_constant_expr(ret) else And(ret, newe)
        #print("\t * * output filter : {}".format(expr_to_sql(ret)))
        return ret
    elif isinstance(op, NoChangeInference) or isinstance(op, ReadTableInference):
        return preds[0]
    elif isinstance(op, InnerJoinInference) or isinstance(op, LeftOuterJoinInference):
        ret = AllAnd(preds[0], preds[1])
        return ret
    elif isinstance(op, GroupByInference):
        # TODO: handle groupby expression
        ret = True
        col_used = set()
        equiv_pairs = {k.get_full_name() if isinstance(k, UnresolvedRelation) else str(k):v for k,v in get_equivalent_pairs(preds[0])}
        for gexpr in op.grouping_expressions:
            if isinstance(gexpr, UnresolvedAttribute): # group column used as it is
                col_used = col_used.union(get_columns_used(gexpr))
        for aexpr in op.aggregation_expressions:
            # scalar aggr can pass predicate, avoid dup with gexpr
            if is_group_expr_scalar_function(aexpr) and not isinstance(aexpr, UnresolvedAttribute): 
                if all([is_col_in_exist_columns(c, set([k for k,v in equiv_pairs.items()])) for c in get_columns_used(aexpr)]):
                    newe = get_filter_replacing_field(aexpr, equiv_pairs)
                    newe = EqualTo(UnresolvedAttribute([get_new_column_name_from_expr(aexpr)]), newe)
                    ret = newe if is_constant_expr(ret) else And(ret, newe)
        if len(col_used) > 0:
            ret = And(ret, get_filter_replacing_nonexist_column(preds[0], col_used))
        return ret
    elif isinstance(op, SubqueryInference):
        # add correlated field to subq table
        correlated_values = {}
        #print("* * * SUBQ: {}".format(SubqueryInference))
        for c in op.correlated_columns:
            correlated_values[c.split('.')[-1]] = op.input_tables[0][0][c]
        subppl_nodes = walk_through_graph(op.subppl, lambda n: [n], [], lambda x,y:x+y)
        #print(" --- correlated values = {}".format(correlated_values))
        for subppl_node in subppl_nodes:
            for tup in subppl_node.inference.input_tables[0]:
                tup.values.update(correlated_values)

        subq_filter = predicate_pushup_pipeline_helper(subppl_nodes)[op.subppl]
        #print("subq_filter = {}".format(expr_to_sql(subq_filter)))
        ## FIXME
        #xx = zeval(subq_filter, op.subppl.inference.input_tables[0][0])
        subq_added = []
        equiv_pairs = get_equivalent_pairs(subq_filter)
        for c in op.correlated_columns:
            for p in equiv_pairs:
                if isinstance(p[1], Literal) and 'orderkey' in c and 'orderkey' in str(p[0]):
                    subq_added.append(EqualTo(UnresolvedAttribute([c]), p[1]))
                if isinstance(p[1], Literal) and 'partkey' in c and 'partkey' in str(p[0]):
                    subq_added.append(EqualTo(UnresolvedAttribute([c]), p[1]))
                if isinstance(p[1], Literal) and 'suppkey' in c and 'suppkey' in str(p[0]):
                    subq_added.append(EqualTo(UnresolvedAttribute([c]), p[1]))
            
        # remove correlated field
        for subppl_node in subppl_nodes:
            for tup in subppl_node.inference.input_tables[0]:
                for col_name,v in correlated_values.items():
                    del tup.values[col_name]

        if len(subq_added) == 0:
            return preds[0]
        subq_added = AllAnd(*subq_added)
        #print("subq ADD : {}".format(expr_to_sql(subq_added)))
        return And(subq_added, preds[0])
    else:
        print(type(op))
        assert(False)


def predicate_pushup_operator(op, input_filters):
    assert(len(input_filters) == len(op.input_tables))
    op.input_filters = input_filters
    candidate = obtain_predicate_candidate_pushup(op, input_filters)
    #print("\t ^ ^ candidate = {}".format(expr_to_sql(candidate)))
    op.output_filter = candidate
    #is_correct = op.verify_correct(check_superset=True)
    is_correct = op.verify_pushup_correct()
    if is_correct:
        return candidate 

def predicate_pushup_pipeline_helper(ppl_nodes):
    queue = []
    processed_pushup_nodes = {}

    for ppl_node in ppl_nodes:
        if isinstance(ppl_node.inference, ReadTableInference):
            queue.append(ppl_node)
            processed_pushup_nodes[ppl_node] = [generate_row_selection_predicate(ppl_node.input_schema[0])]
            ppl_node.row_sel_pred_derived = processed_pushup_nodes[ppl_node][0]

    ret = {}
    while len(queue) > 0:
        ppl_node = queue.pop(0)
        preds_to_push = processed_pushup_nodes[ppl_node]
        #print("push up for {} / {}: ".format(ppl_node.plan_node, ppl_node.inference))
        #print("---input filters: \n{}".format('\n'.join(['\t{}'.format(expr_to_sql(i)) for i in preds_to_push])))
        new_pred = predicate_pushup_operator(ppl_node.inference, preds_to_push)
        #print("---output filter: \n\t{}".format(expr_to_sql(new_pred)))
        ret[ppl_node] = new_pred
        if hasattr(ppl_node, "need_materialization") and ppl_node.need_materialization==True:
            ppl_node.F_i = And(ppl_node.F_i, new_pred)
        for i,downstream_op in enumerate(ppl_node.downstream_nodes):
            if downstream_op in processed_pushup_nodes:
                # TODO: cannot handle union with multiple inputs
                if ppl_node == downstream_op.upstream_nodes[0]:
                    processed_pushup_nodes[downstream_op].insert(0, new_pred)
                else:
                    processed_pushup_nodes[downstream_op].append(new_pred)
            else:
                processed_pushup_nodes[downstream_op] = [new_pred]
            if len(processed_pushup_nodes[downstream_op]) == len(downstream_op.upstream_nodes):
                queue.append(downstream_op)

    return ret

def walk_through_graph(start_node, handle_node, rs_init, union_rs):
    queue = [start_node]
    processed = set()
    while len(queue) > 0:
        ppl_node = queue.pop(0)
        ret_temp = handle_node(ppl_node)
        rs_init = union_rs(ret_temp, rs_init)
        for n in ppl_node.upstream_nodes:
            if n not in processed:
                queue.append(n)
    return rs_init

def walk_graph_recursive(start_node, handle_node, rs_init=None, union_rs=lambda x,y:x):
    queue = [start_node]
    processed = set()
    while len(queue) > 0:
        ppl_node = queue.pop(0)
        ret_temp = handle_node(ppl_node)
        rs_init = union_rs(ret_temp, rs_init)
        if isinstance(ppl_node.inference, SubqueryInference):
            #print("walk sub graph")
            rs_init = walk_graph_recursive(ppl_node.inference.subppl, handle_node, rs_init, union_rs)
            print(rs_init)
        for n in ppl_node.upstream_nodes:
            if n not in processed:
                queue.append(n)
    return rs_init

def lineage_inference_no_intermediate_result(ppl, output_filter):
    readtable_nodes = set()
    ppl_nodes = walk_through_graph(ppl, lambda n: [n], [], lambda x,y:x+y)
    readtable_nodes = set(walk_through_graph(ppl, lambda n: [n] if isinstance(n.inference, ReadTableInference) else [], [], lambda x,y:x+y))
    for n in readtable_nodes:
        n.processing_order = -1
        n.row_sel_pred_derived = None
    #print(" !!! ---- OUTPUT FILTER = {}".format(expr_to_sql(output_filter)))

    diverged_nodes = {} # key: ppl_node, value: new predicate (after pushup) 

    node_source_map = {} # key: ppl_node, value: all readtable node flow to this node
    for n in ppl_nodes:
        related_sources = set()
        queue = [n]
        processed = set()
        while len(queue) > 0:
            ppl_node = queue.pop(0)
            processed.add(ppl_node)
            if isinstance(ppl_node.inference, ReadTableInference):
                related_sources.add(ppl_node)
            for n1 in ppl_node.upstream_nodes:
                if all([x in processed for x in n1.downstream_nodes]):
                    queue.append(n1)
        node_source_map[n] = related_sources

    queue = [ppl]
    processed_pushdown_nodes = {ppl: [output_filter]} # key: node, value: [] of predicates

    # first pushdown
    #print(" * * * * * Phase 1: push down * * * * *")
    while len(queue) > 0:
        ppl_node = queue.pop(0)
        pred_to_push = processed_pushdown_nodes[ppl_node][0] if len(processed_pushdown_nodes[ppl_node]) == 1 else AllOr(*processed_pushdown_nodes[ppl_node])
        
        no_intermediate_result = predicate_pushdown_operator(ppl_node.inference, pred_to_push, False)
        ppl_node.need_materialization = (no_intermediate_result==False)
        ppl_node.F_i = pred_to_push
        # print("pushdo down for {} / {}: ".format(ppl_node.plan_node, ppl_node.inference))
        # print("---output filter: \n\t{}".format(expr_to_sql(ppl_node.inference.output_filter)))
        # print("---input filters: \n{}".format('\n'.join(['\t{}'.format(expr_to_sql(i)) for i in ppl_node.inference.input_filters])))
        # if no_intermediate_result==False:
        #     if ppl_node not in diverged_nodes:
        #         diverged_nodes[ppl_node] = pred_to_push
        #     else:
        #         pass
        
        for i,upstream_op in enumerate(ppl_node.upstream_nodes):
            if upstream_op in processed_pushdown_nodes:
                processed_pushdown_nodes[upstream_op].append(ppl_node.inference.input_filters[i])
            else:
                processed_pushdown_nodes[upstream_op] = [ppl_node.inference.input_filters[i]]
            if len(upstream_op.downstream_nodes) == len(processed_pushdown_nodes[upstream_op]):
                queue.append(upstream_op)
    def set_original_pred(n):
        if isinstance(n.inference, ReadTableInference):
            n.original_predicate = simplify_pred(n.inference.input_filters[0])
    walk_graph_recursive(ppl, set_original_pred)

    # push up for each ReadTable nodes that applies a lineage filter
    #print(" * * * * * Phase 2: push up * * * * *")
    processed_pushup_nodes = predicate_pushup_pipeline_helper(ppl_nodes)
    # for ppl_node,new_pred in processed_pushup_nodes.items():
    #     if ppl_node in diverged_nodes:
    #         diverged_nodes[ppl_node] = And(new_pred, diverged_nodes[ppl_node]) 
    #         print("diverge: {} / {}\n".format(ppl_node.inference, expr_to_sql(new_pred)))
    def set_rowsel_pred(n):
        if isinstance(n.inference, ReadTableInference):
            n.row_sel_pred_derived = n.inference.input_filters[0]
    walk_graph_recursive(ppl, set_rowsel_pred)

    # pushdown again
    #queue = [n for n,p in diverged_nodes.items()]    
    queue = [n for n in ppl_nodes if hasattr(n, 'need_materialization') and n.need_materialization==True]
    #processed_pushdown_nodes = {k:[v] for k,v in diverged_nodes.items()}
    processed_pushdown_nodes = {k:[] for k in queue}
    while len(queue) > 0:
        ppl_node = queue.pop(0)
        # if ppl_node in diverged_nodes:
        #     if len(processed_pushdown_nodes[ppl_node]) > 1:
        #         pred_to_push = And(processed_pushdown_nodes[ppl_node][-1], diverged_nodes[ppl_node])
        #     else:
        #         pred_to_push = diverged_nodes[ppl_node]
        # else:
        if hasattr(ppl_node,'need_materialization') and ppl_node.need_materialization==True and hasattr(ppl_node, 'F_i'):
            if len(processed_pushdown_nodes[ppl_node]) >= 1:
                pred_to_push = And(processed_pushdown_nodes[ppl_node][-1], ppl_node.F_i)
            else:
                pred_to_push = ppl_node.F_i
        else:
            pred_to_push = processed_pushdown_nodes[ppl_node][0] if len(processed_pushdown_nodes[ppl_node]) == 1 else AllOr(*processed_pushdown_nodes[ppl_node])
        pred_to_push = simplify_conjunction(simplify_pred(pred_to_push))

        no_intermediate_result = predicate_pushdown_operator(ppl_node.inference, pred_to_push, False)
        print("\npushdown for {} / {}: ".format(ppl_node.plan_node, ppl_node.inference))
        print("---output filter: \n\t{}".format(expr_to_sql(ppl_node.inference.output_filter)))
        print("---input filters: \n{}".format('\n'.join(['\t{}'.format(expr_to_sql(i)) for i in ppl_node.inference.input_filters])))
        
        for i,upstream_op in enumerate(ppl_node.upstream_nodes):
            if upstream_op in processed_pushdown_nodes:
                processed_pushdown_nodes[upstream_op].append(ppl_node.inference.input_filters[i])
            else:
                processed_pushdown_nodes[upstream_op] = [ppl_node.inference.input_filters[i]]
            if len(upstream_op.downstream_nodes) == len(processed_pushdown_nodes[upstream_op]):
                queue.append(upstream_op)
            
    return ppl_nodes

def predicate_pushup_pipeline(ppl, input_table_filters):
    ppl_nodes = walk_through_graph(ppl, lambda n: [n], [], lambda x,y:x+y)
    queue = []
    processed_pushup_nodes = {}
    for ppl_node in ppl_nodes:
        if isinstance(ppl_node.inference, ReadTableInference):
            queue.append(ppl_node)
            if ppl_node.plan_node.tableIdentifier in input_table_filters:
                processed_pushup_nodes[ppl_node] = [input_table_filters[ppl_node.plan_node.tableIdentifier]]
            else:
                processed_pushup_nodes[ppl_node] = [True]

    while len(queue) > 0:
        ppl_node = queue.pop(0)
        preds_to_push = processed_pushup_nodes[ppl_node]
        new_pred = predicate_pushup_operator(ppl_node.inference, preds_to_push)
        # print("push up for {} / {}: ".format(ppl_node.plan_node, ppl_node.inference))
        # print("---input filters: \n{}".format('\n'.join(['\t{}'.format(expr_to_sql(i)) for i in ppl_node.inference.input_filters])))
        # print("---output filter: \n\t{}".format(expr_to_sql(new_pred)))
        for i,downstream_op in enumerate(ppl_node.downstream_nodes):
            if downstream_op in processed_pushup_nodes:
                # TODO: cannot handle union with multiple inputs
                if ppl_node == downstream_op.upstream_nodes[0]:
                    processed_pushup_nodes[downstream_op].insert(0, new_pred)
                else:
                    processed_pushup_nodes[downstream_op].append(new_pred)
            else:
                processed_pushup_nodes[downstream_op] = [new_pred]
            if len(processed_pushup_nodes[downstream_op]) == len(downstream_op.upstream_nodes):
                queue.append(downstream_op)
    return simplify_pred(ppl_nodes[-1].inference.output_filter)


import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import psycopg2
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)

def psql_run_query(pgconn, sql, include_schema=False):
    cursor = pgconn.cursor()
    cursor.execute(sql)
    rs = cursor.fetchall()
    if include_schema:
        schema = [desc[0] for desc in cursor.description]
        return rs, schema
    else:
        return rs
def convert_rs_to_rowsel_variable(rs, schema, rowsel_pred):
    equiv_pairs = get_equivalent_pairs(rowsel_pred)
    rowsel_variable_to_column = [(p2.value,p1.get_column_name()) for p1,p2 in equiv_pairs]
    rowsel_variable_values = {k:[] for k,v in rowsel_variable_to_column}
    for i in range(len(schema)):
        rowsel_variable = list(filter(lambda pair: pair[1]==schema[i], rowsel_variable_to_column))
        if len(rowsel_variable) == 0:
            #print("Column {} not exist in rowsel".format(schema[i]))
            continue
        rowsel_variable = rowsel_variable[0][0]
        rowsel_variable_values[rowsel_variable] = set([row[i] for row in rs])
    return rowsel_variable_values

def get_projection(n):
    return [col_name for col_name, col_type in n.input_schema[0] if 'key' in col_name]


def clean_sql(sql):
    return sql.replace("extract('year',l_shipdate)", "extract(year from l_shipdate)").replace("extract('year',o_orderdate)", 'extract(year from o_orderdate)').replace('l1.','')

import time
def retrieve_lineage_data(ppl, pgconn):
    start = time.time()
    readtable_nodes = walk_graph_recursive(ppl, \
                         lambda n: [n] if isinstance(n.inference, ReadTableInference) else [],
                         rs_init=[], union_rs=lambda x,y:x+y)
    print(ppl.plan_node)
    for n in readtable_nodes:
        print("Read {}".format(n.plan_node.tableIdentifier))
    rowsel_variable_values = {}
    round = 0
    lastround_value_count = {}
    table_count = {}
    while round==0 or any([k not in lastround_value_count or len(v)!=lastround_value_count[k] for k,v in rowsel_variable_values.items()]):
        print("\n=====round {}======".format(round))
        lastround_value_count = {k:len(v) for k,v in rowsel_variable_values.items()}
        print("\tround cnt = {}".format('\n'.join(['\t{}={}'.format(k,v) for k,v in lastround_value_count.items() if 'key' in str(k)])))
        for n in readtable_nodes:
            if round == 0:
                pred = n.original_predicate
            else:
                simplified_p = simplify_pred(simplify_pred_remove_selfrow(n.inference.input_filters[0]))
                pred = replace_rowsel_variables(simplified_p, rowsel_variable_values)
            if round > 0 and type(pred) is bool: # no pred is derived, skip
                continue
            sql = "SELECT {} FROM {} WHERE {}".format(','.join(get_projection(n)),\
                                                    n.plan_node.tableIdentifier,\
                                                    expr_to_sql(pred))
            sql = clean_sql(sql)
            if round == 0 and type(pred) is bool: # a small optimization to not run True pred
                sql = "{} LIMIT 1".format(sql)
            rs, schema = psql_run_query(pgconn, sql, include_schema=True)
            print("table {} -- {}".format(n.plan_node.tableIdentifier, len(rs)))
            print("\t{}".format(expr_to_sql(n.original_predicate if round==0 else simplified_p)[:1000]))
            #print("\telapse: {} sec".format(time.time() - start))
            table_count[n] = len(rs)
            temp = convert_rs_to_rowsel_variable(rs, schema, n.row_sel_pred_derived)
            if round == 0 and type(pred) is bool:
                temp = {k:[] for k,v in temp.items()}
            rowsel_variable_values.update(temp)
        round += 1

    print("elapse: {}".format(time.time() - start))
    for k,v in table_count.items():
        print("Table {} return {} rows".format(k.plan_node.tableIdentifier,v))
        
            
    
