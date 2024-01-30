import os
import sys
from verify_core_op import *
from ir import *
from z3_eval_plan_ir import *

def obtain_predicate_candidate(op, pred):
    candidates = []
    if isinstance(op, FilterInference):
        # for subquery, replace unhandled correlated columns
        exist_columns = set([col for col,v in op.input_tables[0][0].values.items()])
        candidates.append(get_filter_replacing_nonexist_column(And(pred, op.condition), exist_columns))
        #candidates.append(get_filter_replacing_nonexist_column(pred, exist_columns))
    elif isinstance(op, ProjectionInference):
        column_to_replace = {k.get_column_name():v for k,v in op.col_func_pair}
        new_pred = get_filter_replacing_field(pred, column_to_replace)
        candidates.append(new_pred)
    elif isinstance(op, NoChangeInference) or isinstance(op, ReadTableInference):
        candidates.append(pred)
    elif isinstance(op, InnerJoinInference) or isinstance(op, LeftOuterJoinInference):
        exist_columns_left = set([col for col,v in op.input_tables[0][0].values.items()])
        exist_columns_right = set([col for col,v in op.input_tables[1][0].values.items()])
        equivalent_pairs = get_equivalent_pairs(pred)
        #pred_to_add_left = get_additional_equivalence_from_equiv_pairs2(equivalent_pairs, exist_columns_left)
        #pred_to_add_right = get_additional_equivalence_from_equiv_pairs2(equivalent_pairs, exist_columns_right)
        target_cols_left = [c for c in op.merge_cols_left]
        target_cols_right = [c for c in op.merge_cols_right]
        for p in equivalent_pairs:
            if isinstance(p[0], UnresolvedAttribute) and isinstance(p[1], UnresolvedAttribute):
                if is_col_in_exist_columns(p[0].get_full_name(),  exist_columns_left) and is_col_in_exist_columns(p[1].get_full_name(),  exist_columns_right):
                    target_cols_left.append(p[0].get_full_name())
                    target_cols_right.append(p[1].get_full_name())
                if is_col_in_exist_columns(p[1].get_full_name(),  exist_columns_left) and is_col_in_exist_columns(p[0].get_full_name(),  exist_columns_right):
                    target_cols_left.append(p[1].get_full_name())
                    target_cols_right.append(p[0].get_full_name())
        equivalent_pairs = equivalent_pairs + [(wrap_column_name(op.merge_cols_left[i]), wrap_column_name(op.merge_cols_right[i])) for i in range(len(op.merge_cols_left))]
        pred_to_add_left = get_additional_equivalence_from_equiv_pairs(equivalent_pairs, target_cols_left, exist_columns_left)
        pred_to_add_right = get_additional_equivalence_from_equiv_pairs(equivalent_pairs, target_cols_right, exist_columns_right)
        # print("joining {} AND {}".format(op.merge_cols_left[0], op.merge_cols_right[0]))
        # print("$ $ equiv pairs = {}".format('\n'.join(['\t{}={}'.format(expr_to_sql(p1),expr_to_sql(p2)) for p1,p2 in equivalent_pairs])))
        # print(" $ $ pred to add left = {}".format(','.join([expr_to_sql(x) for x in pred_to_add_left])))
        # print(" $ $ pred to add right = {}".format(','.join([expr_to_sql(x) for x in pred_to_add_right])))
        # print("")

        #pred_left = get_filter_replacing_field(pred, {op.merge_cols_right[i]:wrap_column_name(op.merge_cols_left[i]) for i in range(len(op.merge_cols_left))})
        pred_left = get_filter_replacing_nonexist_column(pred, exist_columns_left)
        pred_left = AllAnd(*([pred_left]+pred_to_add_left))
        #pred_right = get_filter_replacing_field(pred, {op.merge_cols_left[i]:wrap_column_name(op.merge_cols_right[i]) for i in range(len(op.merge_cols_left))})
        pred_right = get_filter_replacing_nonexist_column(pred, exist_columns_right)
        pred_right = AllAnd(*([pred_right]+pred_to_add_right))
        candidates.append([pred_left, pred_right])
    elif isinstance(op, PivotInference):
        cols = [c.get_full_name() for c in op.group_columns]
        if any([is_col_in_exist_columns(c, cols) for c in get_columns_used(pred)]):
            remainig_col_pred = get_filter_replacing_nonexist_column(pred, cols)
            candidates.append(remainig_col_pred)
        else:
            assert(False)

    elif isinstance(op, GroupByInference):
        column_to_replace = {}
        for expr in op.aggregation_expressions:
            if is_group_expr_scalar_function(expr):
                column_to_replace[get_new_column_name_from_expr(expr)] = expr
        exist_columns = set([col for col,v in op.input_tables[0][0].values.items()])
        pred1 = get_filter_replacing_field(pred, column_to_replace)
        pred1 = get_filter_replacing_nonexist_column(pred1, exist_columns)
        candidates.append(pred1)

        for expr in op.aggregation_expressions:
            if len(get_columns_used(expr)) == 0 or is_group_expr_scalar_function(expr):
                continue
            column_to_replace[get_new_column_name_from_expr(expr)] = wrap_column_name(get_columns_used(expr)[0])
        pred2 = get_filter_replacing_field(pred, column_to_replace)
        candidates.append(pred2)
        #print("candidates = {}".format('\n'.join([expr_to_sql(e) for e in candidates])))
    
    elif isinstance(op, SubqueryInference):
        # pushdown to other table
        candidates.append(And(pred, op.condition))
    elif isinstance(op, UnpivotInference):
        columns_used = get_columns_used(pred)
        if any([is_col_in_exist_columns(c, op.id_vars) for c in columns_used]):
            new_pred = get_filter_replacing_nonexist_column(pred, op.id_vars)
            candidates.append(new_pred)
        else:
            #get_filter_replacing_compare()
            assert(False) # FIXME: unhandled for now
            pass

    else:
        print(type(op))
        assert(False)
    
    return candidates

def obtain_subquery_filter(op, pred):
    # find equivalence pairs in pred
    equiv_pairs = list(filter(lambda pair: isinstance(pair[0], UnresolvedAttribute) and isinstance(pair[1], Literal), get_equivalent_pairs(pred)))
    # replace all fields from left table with equiv pair 
    field_to_replace = {pair[0].get_full_name(): pair[1] for pair in equiv_pairs}
    #print("* * * --- subquery field to replace: {}".format(field_to_replace))
    #print(expr_to_sql(pred))

    pred_to_push = True
    if isinstance(op.subquery_cmp, InSubquery) or isinstance(op, EqualTo):
        field = op.subquery_cmp.field if isinstance(op.subquery_cmp, InSubquery) else op.subquery_cmp.left
        print("field = {}, {}".format(field, is_col_in_exist_columns(field.get_full_name(), [p1.get_full_name() for p1,p2 in equiv_pairs])))
        if isinstance(field, UnresolvedAttribute) and is_col_in_exist_columns(field.get_full_name(), [p1.get_full_name() for p1,p2 in equiv_pairs]):
            target_cmp = list(filter(lambda x: is_col_in_exist_columns(field.get_column_name(), [x[0].get_full_name()]), equiv_pairs))
            for k1,v1 in target_cmp:
                pred_to_push = And(pred_to_push, EqualTo(wrap_column_name(op.subppl.output_schema[0][0]), v1))#EqualTo(wrap_column_name(op.subppl.output_schema[0][0]), target_cmp[1])
            print(expr_to_sql(pred_to_push))
    # TODO: InSubquery, propagate outer field
    def replace_variable_in_correlated_subquery(ppl_node):
        if isinstance(ppl_node.inference, FilterInference):
            to_replace = {}
            for k,v in field_to_replace.items():
                if is_col_in_exist_columns(k, op.correlated_columns):
                    to_replace[k] = v
            ppl_node.inference.condition = get_filter_replacing_field(ppl_node.plan_node.condition, to_replace, True)
            ppl_node.inference.condition = get_filter_replacing_nonexist_column(ppl_node.inference.condition, set([col for col,v in ppl_node.inference.input_tables[0][0].values.items()]))
        return None
    op.subppl.iterate_pipeline_node(replace_variable_in_correlated_subquery, union_rs=lambda x,y:None)
    
    #print(" _________ * * Predicate push through subpip = {}".format(expr_to_sql(pred_to_push)))
    return pred_to_push
    # print("+ + subpipeline")
    # print(" + + \t{}".format(subppl.plan_node))
    # print(" + + \t+ + {}".format(subppl.upstream_nodes[0].inference.condition))


# can_store_intermediate_result can be ignored
# just used for paper evaluation as a baseline (no optimization)
def predicate_pushdown_operator(op, output_filter, can_store_intermediate_result=True):
    op.output_filter = output_filter
    if isinstance(op, GroupByInference):
        # predicate rewrite, remove aggr columns
        col_used = set()
        for gexpr in op.grouping_expressions:
            col_used = col_used.union(get_columns_used(gexpr))
        op.output_filte = get_filter_replacing_nonexist_column(output_filter, col_used)
    if isinstance(op, SubqueryInference):
        print("\t\t-------subquery begin---------")
        subquery_filter = obtain_subquery_filter(op, output_filter)
        op.ppl_nodes = predicate_pushdown_pipeline(op.subppl, subquery_filter, can_store_intermediate_result)
        print("\t\t-------subquery end---------")
        exist_columns = set([col for col,v in op.input_tables[0][0].values.items()])
        op.condition = get_filter_replacing_nonexist_column(op.condition, exist_columns)

    for candidate in obtain_predicate_candidate(op, output_filter):
        #print("candidate = {}".format(candidate))
        op.input_filters = candidate if type(candidate) is list else [candidate]
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


def predicate_pushdown_pipeline(ppl, output_filter, can_store_intermediate_result=True): # ppl: the last node
    ppl_nodes = []
    queue = [ppl]
    processed_pushdown_nodes = {ppl: [output_filter]} # key: node, value: [] of predicates
    #print("output filter = {}".format(expr_to_sql(output_filter)))
    while len(queue) > 0:
        ppl_node = queue.pop(0)
        ppl_nodes.append(ppl_node)
        pred_to_push = processed_pushdown_nodes[ppl_node][0] if len(processed_pushdown_nodes[ppl_node]) == 1 else AllOr(*processed_pushdown_nodes[ppl_node])
        if hasattr(ppl_node, 'need_materialization') and ppl_node.need_materialization==True and hasattr(ppl_node, 'F_i'):
            pred_to_push = And(ppl_node.F_i, pred_to_push)

        no_intermediate_result = predicate_pushdown_operator(ppl_node.inference, pred_to_push, can_store_intermediate_result)
        ppl_node.need_materialization = (no_intermediate_result==False)
        ppl_node.F_i = pred_to_push
        # print("pushdo down for {} / {}: ".format(ppl_node.plan_node, ppl_node.inference))
        # print("---output filter: \n\t{}".format(expr_to_sql(ppl_node.inference.output_filter)))
        # print("---input filters: \n{}".format('\n'.join(['\t{}'.format(expr_to_sql(i)) for i in ppl_node.inference.input_filters])))
        if no_intermediate_result==False and can_store_intermediate_result:
            ppl_node.original_predicate = pred_to_push
            # column projection optimization, here customized for TPC-H
            #opt_schema = list(filter(lambda pair: pair[0].endswith('key'), ppl_node.output_schema))
            #pred_to_push = generate_row_selection_predicate(opt_schema)
            pred_to_push = generate_row_selection_predicate(ppl_node.output_schema)
            ppl_node.intermediate_result_row_sel_pred = pred_to_push
            #print("* * * * New predicate = {}".format(pred_to_push))
            predicate_pushdown_operator(ppl_node.inference, pred_to_push)
        
        for i,upstream_op in enumerate(ppl_node.upstream_nodes):
            if upstream_op in ppl_nodes:
                continue
            if upstream_op in processed_pushdown_nodes:
                processed_pushdown_nodes[upstream_op].append(ppl_node.inference.input_filters[i])
            else:
                processed_pushdown_nodes[upstream_op] = [ppl_node.inference.input_filters[i]]
            if len(upstream_op.downstream_nodes) == len(processed_pushdown_nodes[upstream_op]):
                queue.append(upstream_op)
    #print("ppl_nodes:")
    return ppl_nodes



def print_filters_on_input_table(ppl_nodes, subquery_level=0):
    indent = ''.join(['\t' for i in range(subquery_level)])
    if subquery_level == 0:
        print("\n========================")
        print("Final pushed filters on the input tables:\n")
    else:
        print("{}----subquery:-----\n".format(indent))
    for n in ppl_nodes:
        if isinstance(n.inference, ReadTableInference):
            if hasattr(n, 'processing_order'):
                print("{}table {}{}: {}\n".format(indent, n.plan_node.tableIdentifier, \
                                                '(process at round {}{})'.format(n.processing_order, ', derive row-sel pred' if hasattr(n, 'row_sel_pred_derived') else ''),\
                                                expr_to_sql(simplify_pred(simplify_pred_remove_selfrow(n.inference.input_filters[0])))))
#                                                expr_to_sql(simplify_pred(n.inference.input_filters[0]))))
            else:
                print("{}table {}: {}\n".format(indent, n.plan_node.tableIdentifier, \
                                                expr_to_sql(simplify_pred(n.inference.input_filters[0]))))
        if isinstance(n.inference, SubqueryInference):
            print_filters_on_input_table(n.inference.ppl_nodes, subquery_level=subquery_level+1)
    for n in ppl_nodes:
        if hasattr(n, 'intermediate_result_row_sel_pred'):
            print("{}* Save intermediate result after {}".format(indent, n.plan_node))