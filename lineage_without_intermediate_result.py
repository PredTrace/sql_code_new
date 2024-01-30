import os
import sys
from verify_core_op import *
from ir import *
from z3_eval_plan_ir import *
from predicate_pushdown import predicate_pushdown_operator, obtain_subquery_filter, obtain_predicate_candidate



# pred is predicate on input
def obtain_predicate_candidate_pushup(op, preds):
    if isinstance(op, FilterInference):
        return preds[0]
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
        equiv_pairs = {k.get_full_name():v for k,v in get_equivalent_pairs(preds[0])}
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
        return preds[0]
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
            

def lineage_inference_no_intermediate_result_operator(op, output_filter):
    op.output_filter = output_filter
    if isinstance(op, GroupByInference):
        # predicate rewrite, remove aggr columns
        col_used = set()
        for gexpr in op.grouping_expressions:
            col_used = col_used.union(get_columns_used(gexpr))
        op.output_filte = get_filter_replacing_nonexist_column(output_filter, col_used)
    if isinstance(op, SubqueryInference):
        #print("\t\t-------subquery begin---------")
        subquery_filter = obtain_subquery_filter(op, output_filter)
        op.ppl_nodes = lineage_inference_no_intermediate_result(op.subppl, subquery_filter)
        #print("\t\t-------subquery end---------")
        exist_columns = set([col for col,v in op.input_tables[0][0].values.items()])
        op.condition = get_filter_replacing_nonexist_column(op.condition, exist_columns)

    for candidate in obtain_predicate_candidate(op, output_filter):
        op.input_filters = candidate if type(candidate) is list else [candidate]
        #print("candidate = {}".format('\n'.join([expr_to_sql(c) for c in op.input_filters])))
        is_small_model = op.check_small_model(check_superset=True)
        if is_small_model:
            is_exact_lineage = op.lineage_exact()
            if any([x==False for x in is_exact_lineage]):
                return False
            is_correct = op.verify_correct(check_superset=True)
            if is_correct:
                return 
            print(op)
            exit(0)
    op.input_filters = [True for i in range(len(op.input_tables))]
    return True

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

def lineage_inference_no_intermediate_result(ppl, output_filter):
    ppl_nodes = []
    readtable_nodes = set()
    ppl_nodes = walk_through_graph(ppl, lambda n: [n], [], lambda x,y:x+y)
    readtable_nodes = set(walk_through_graph(ppl, lambda n: [n] if isinstance(n.inference, ReadTableInference) else [], [], lambda x,y:x+y))
    for n in readtable_nodes:
        n.processing_order = -1
        n.row_sel_pred_derived = None
    #print(" !!! ---- OUTPUT FILTER = {}".format(expr_to_sql(output_filter)))

    table_filters = {} # key: readtable node; value: pushed down filter on that table
    round = 0
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

    while True:
        round += 1
        queue = [ppl]
        processed_pushdown_nodes = {ppl: [output_filter]} # key: node, value: [] of predicates
        make_progress = False
        subquery_unhandled = False
        
        # first pushdown
        #print(" * * * * * Phase 1: push down * * * * *")
        while len(queue) > 0:
            ppl_node = queue.pop(0)
            if ppl_node in diverged_nodes:
                pred_to_push = diverged_nodes[ppl_node]
            else:
                pred_to_push = processed_pushdown_nodes[ppl_node][0] if len(processed_pushdown_nodes[ppl_node]) == 1 else AllOr(*processed_pushdown_nodes[ppl_node])
            
            pred_to_push = simplify_pred(pred_to_push)
            if isinstance(ppl_node.inference, ReadTableInference):
                make_progress = True # if no more table can be inferred, mark as no progress or
                if not hasattr(ppl_node, 'processing_order') or ppl_node.processing_order == -1:
                    #print("set {} = {}".format(ppl_node.plan_node.tableIdentifier, expr_to_sql(pred_to_push)))
                    table_filters[ppl_node] = pred_to_push
                    ppl_node.processing_order = round

            no_intermediate_result = lineage_inference_no_intermediate_result_operator(ppl_node.inference, pred_to_push)
            # print("pushdo down for {} : ".format(ppl_node.inference))
            # print("---output filter: \n\t{}".format(expr_to_sql(ppl_node.inference.output_filter)))
            # print("---input filters: \n{}".format('\n'.join(['\t{}'.format(expr_to_sql(i)) for i in ppl_node.inference.input_filters])))
            pushdown_holds = []
            if no_intermediate_result==False:
                if ppl_node not in diverged_nodes:
                    diverged_nodes[ppl_node] = pred_to_push
                else:
                    pass
                #print("# # #   diverge at {}".format(ppl_node.plan_node))
                #print("---output filter: \n\t{}".format(expr_to_sql(ppl_node.inference.output_filter)))
                #print("---input filters: \n{}".format('\n'.join(['\t{}'.format(expr_to_sql(simplify_pred(i))) for i in ppl_node.inference.input_filters])))
                if len(ppl_node.upstream_nodes) > 1:
                    # some heuristic to choose which path to pushdown (then up), for paths whose sources hasn't been processed
                    # decision should depend on some cardinality estimation: TODO
                    # now push down through the first path
                    temp0 = [(i, len(get_columns_used(ppl_node.inference.input_filters[i]))) for i in range(0, len(ppl_node.inference.input_filters))]
                    temp = [(i, temp0[i][1] if not all([source in table_filters for source in node_source_map[ppl_node.upstream_nodes[i]]]) else -1)  for i in range(0, len(ppl_node.inference.input_filters))]
                    max_fields = max([x[1] for x in temp])
                    keep = 0
                    #print("temp0 = {}, temp = {}, max_fields = {}".format(temp0, temp, max_fields))
                    for i in range(0, len(ppl_node.inference.input_filters)):
                        if temp[i][1] == max_fields:
                            keep = i
                            #break
                    for i in range(0, len(ppl_node.inference.input_filters)):
                        if i != keep:
                            pushdown_holds.append(i)
                    #print("pushdown holds = {}".format(pushdown_holds))
                elif isinstance(ppl_node.inference, SubqueryInference):
                    subquery_unhandled = True
                
            for i,upstream_op in enumerate(ppl_node.upstream_nodes):
                if i in pushdown_holds: # do not move on to this path
                    continue
                if upstream_op in processed_pushdown_nodes:
                    processed_pushdown_nodes[upstream_op].append(ppl_node.inference.input_filters[i])
                else:
                    processed_pushdown_nodes[upstream_op] = [ppl_node.inference.input_filters[i]]
                if len(upstream_op.downstream_nodes) == len(processed_pushdown_nodes[upstream_op]):
                    queue.append(upstream_op)

        #print(" len table filter = {}".format(len(table_filters)))
        if subquery_unhandled==False and (len(table_filters) == len(readtable_nodes) or make_progress == False):
            break
        if round > len(ppl_nodes):
            assert(False)

        # push up for each ReadTable nodes that applies a lineage filter
        #print(" * * * * * Phase 2: push up * * * * *")
        queue = []
        processed_pushup_nodes = {}
        for ppl_node in ppl_nodes:
            if isinstance(ppl_node.inference, ReadTableInference):
                queue.append(ppl_node)
                if hasattr(ppl_node, 'row_sel_pred_derived') and ppl_node.row_sel_pred_derived is not None:
                    processed_pushup_nodes[ppl_node] = [ppl_node.row_sel_pred_derived]
                elif ppl_node in processed_pushdown_nodes:
                    processed_pushup_nodes[ppl_node] = [generate_row_selection_predicate(ppl_node.input_schema[0])]
                    ppl_node.row_sel_pred_derived = processed_pushup_nodes[ppl_node][0]
                else:
                    processed_pushup_nodes[ppl_node] = [True]

        while len(queue) > 0:
            ppl_node = queue.pop(0)
            preds_to_push = processed_pushup_nodes[ppl_node]
            new_pred = predicate_pushup_operator(ppl_node.inference, preds_to_push)
            # print("push up for {} / {}: ".format(ppl_node.plan_node, ppl_node.inference))
            # print("---input filters: \n{}".format('\n'.join(['\t{}'.format(expr_to_sql(i)) for i in ppl_node.inference.input_filters])))
            # print("---output filter: \n\t{}".format(expr_to_sql(new_pred)))
            if ppl_node in diverged_nodes:
                diverged_nodes[ppl_node] = And(new_pred, diverged_nodes[ppl_node]) 
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

        #print("\n============END ROUND {}=============\n".format(round))

    # reset predicate for ReadTable nodes that are processed first
    for ppl_node,pred in table_filters.items():
        ppl_node.inference.input_filters = [pred]
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