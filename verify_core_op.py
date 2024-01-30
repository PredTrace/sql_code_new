import z3
import os
import sys
from util import *
from z3_eval_plan_ir import *
from ir import And,Or


class PipelineNode(object):
    def __init__(self, plan_node_ref):
        self.upstream_nodes = []
        self.downstream_nodes = []
        self.plan_node = plan_node_ref
        self.inference = None
        self.is_materialized = False # result materialized?
        self.input_schema = []
        self.output_schema = None
        # dynamic fields:
        # row_sel_pred_derived
        # processing_order 
        # intermediate_result_row_sel_pred
        self.processing_order = -1
    def add_upstream_node(self, ppl_node):
        self.upstream_nodes.append(ppl_node)
        ppl_node.downstream_nodes.append(self)
    def iterate_pipeline_node(self, f, union_rs):
        ret = f(self)
        for n in self.upstream_nodes:
            ret = union_rs(ret, n.iterate_pipeline_node(f, union_rs))
        return ret


class OperatorInference(object):

    def check_small_model(self, check_superset=False):
        return True
    def add_input_constraint(self, c):
        if self.input_constraint is None:
            self.input_constraint = c
        else:
            self.input_constraint = z3.And(self.input_constraint, c)
    def run_output_filter(self, output):
        ret = []
        for t in output:
            output_t = Tuple(t.values, And(t.exist_cond, self.output_filter), t.count)
            ret.append(output_t)
        return ret
    def run_input_filter(self, inputs, input_filters):
        ret = []
        for i,input_t in enumerate(inputs):
            r = []
            for t in input_t:
                cond = input_filters[i]
                new_input_t = Tuple(t.values, And(t.exist_cond, cond), t.count)
                r.append(new_input_t)
            ret.append(r)
        return ret
    def get_all_table_variables(self, include_null=False):
        vs = []
        for table in self.input_tables:
            for t in table:
                vs.extend([getv(v1) for k1,v1 in t.values.items()])
                if include_null:
                    vs.extend([v1.isnull for k1,v1 in t.values.items()])
        return vs
    def verify_pushup_correct(self): # check_superset=True
        rs_temp = self.run_input_filter(self.input_tables, self.input_filters)
        for table in rs_temp:
            for tup in table:
                tup.exist_cond = tup.eval_exist_cond()
        rs1 = self.run_operator(rs_temp)
        rs2 = self.run_output_filter(rs1)
        #print("G(T) exist cond = {}".format(z3_simplify(rs1[0].eval_exist_cond())))
        #print("F(G(T)) exist cond = {}".format(z3_simplify(rs2[0].eval_exist_cond())))
        expr = z3.And(*[z3.Implies(rs2[i].eval_exist_cond(), rs1[i].eval_exist_cond()) for i in range(len(rs1))])
        return check_always_hold(expr)

    def verify_correct(self, check_superset=False):
        #print("Run f(op())")
        rs1_temp = self.run_operator(self.input_tables)
        #print(rs1_temp[0])
        rs1 = self.run_output_filter(rs1_temp)
        #print(rs1[0])
        #print("----")
        #print("")
        #print("Run op(g())")
        #print("-- input filter 0 = {}".format(self.input_filters[0]))
        rs2_temp = self.run_input_filter(self.input_tables, self.input_filters)
        for table in rs2_temp:
            for tup in table:
                tup.exist_cond = tup.eval_exist_cond()
        #print(rs2_temp[0][0])
        rs2 = self.run_operator(rs2_temp)
        if check_superset:
            rs2 = self.run_output_filter(rs2)
        #print(rs2[0])
        # print(rs1[-1].exist_cond)
        # print(rs1[-1].values)
        #print("rs1 = {} / {} / {}".format(rs1[-1].values, rs1[-1].exist_cond, z3.simplify(rs1[-1].eval_exist_cond())))
        #print("rs2 = {} / {} / {}".format(rs2[-1].values, rs2[-1].exist_cond, z3.simplify(rs2[-1].eval_exist_cond())))
        # print(rs1[-1].exist_cond)
        # print("---- {}".format(z3.simplify(rs1[-1].eval_exist_cond())))
        # print("---- {}".format(z3.simplify(rs2[-1].eval_exist_cond())))
        # print("input filter = {}".format(self.input_filters[0]))
        # print("output filter = {}".format(self.output_filter))

        if len(rs1) == len(rs2):
            # only checking exist but not tuple content --> used only for filter and inner join
            #                                               other operators need to rewrite this
            # print(rs1[0].eval_exist_cond())
            # print(rs2[0].eval_exist_cond())
            expr = z3.And(*[rs1[i].eval_exist_cond() == rs2[i].eval_exist_cond() for i in range(len(rs1))])
            #expr = z3.And(*[rs1[i].eval_exist_cond() == rs2[i].eval_exist_cond() for i in range(2,3)])
            #vs = self.get_all_table_variables()
            if self.input_constraint is None:
                return check_always_hold(expr)
            else:
                return check_always_hold(z3.Implies(self.input_constraint, expr))
        else:
            assert(False, "TODO: CANNOT VERIFY FOR DIFFERENT NUMBER OF TUPLES")
    def output_exists(self):
        rs1_temp = self.run_operator(self.input_tables)
        rs1 = self.run_output_filter(rs1_temp)
        self.input_constraint = z3.Or(*[tup.eval_exist_cond()==True for tup in rs1])
        return self.input_constraint
    def lineage_exact(self):
        return [True for i in range(len(self.input_tables))]
    
    
class ReadTableInference(OperatorInference):
    def __init__(self, input_table, output_filter):
        self.output_filter = output_filter
        self.input_tables = [input_table]
        self.input_constraint = None
    def verify_correct(self, check_superset=False):
        return True
    def run_operator(self, input_tables):
        return input_tables[0]
    
class FilterInference(OperatorInference):
    def __init__(self, input_table, condition, output_filter):
        self.input_tables = [input_table]
        self.condition = condition
        self.output_filter = output_filter
        #self.input_filters = [And(self.condition, self.output_filter)]
        self.input_constraint = None
    def run_operator(self, input_tables):
        output = []
        for t in input_tables[0]:
            output_t = Tuple(t.values, getv(zeval(And(t.exist_cond, self.condition), t)), t.count)
            output.append(output_t)
        return output

class InnerJoinInference(OperatorInference):
    def __init__(self, table_left, table_right, cols_left, cols_right, output_filter):
        self.input_tables = [table_left, table_right]
        self.merge_cols_left = cols_left if type(cols_left) is list else [cols_left]
        self.merge_cols_right = cols_right if type(cols_right) is list else [cols_right]     
        self.output_filter = output_filter
        self.rename_left = {k:k for k,v in self.input_tables[0][0].values.items()}
        self.rename_right = {k:k for k,v in self.input_tables[1][0].values.items()}
        for col_name_l,v in self.input_tables[0][0].values.items():
            for col_name_r,v in self.input_tables[1][0].values.items():
                if col_name_l == col_name_r and col_name_l not in self.merge_cols_left and col_name_r not in self.merge_cols_right:
                   self.rename_left[col_name_l] = '{}_x'.format(col_name_l)
                   self.rename_right[col_name_r] = '{}_y'.format(col_name_r)
        self.input_constraint = None

    def run_operator(self, input_tables):
        table_left = input_tables[0]
        table_right = input_tables[1]
        output = []
        for left_t in table_left:
            for right_t in table_right:
                temp = {self.rename_left[k] if k in self.rename_left else k:v for k,v in left_t.values.items()}
                temp.update( {self.rename_right[k] if k in self.rename_left else k:v for k,v in right_t.values.items()})
                join_cond = z3.And(*[left_t[self.merge_cols_left[i]] == right_t[self.merge_cols_right[i]] for i in range(len(self.merge_cols_left))]) # inner join :if merge cols left and right are different, might have bugs.
                t = Tuple(temp, z3.And(left_t.eval_exist_cond(), right_t.eval_exist_cond(), join_cond), left_t.count*right_t.count)
                output.append(t)
        return output
    def lineage_exact(self):
        equiv_pairs = get_equivalent_pairs(self.input_filters[0])
        equiv_pairs = equiv_pairs + get_equivalent_pairs(self.input_filters[1])
        cols_match_value = [p1.get_full_name() if isinstance(p1, UnresolvedAttribute) and isinstance(p2, Literal) else None for p1,p2 in equiv_pairs]
        cols_match_value += [p2.get_full_name() if isinstance(p2, UnresolvedAttribute) and isinstance(p1, Literal) else None for p1,p2 in equiv_pairs]
        cols_match_value = set(list(filter(lambda x: x is not None, cols_match_value)))
        if all([is_col_in_exist_columns(c, cols_match_value) for c in self.merge_cols_left]):
            return [True, True]
        if all([is_col_in_exist_columns(c, cols_match_value) for c in self.merge_cols_right]):
            return [True, True]
        return [False, False]
    


class LeftOuterJoinInference(OperatorInference):  
    def __init__(self, table_left, table_right, cols_left, cols_right, output_filter):
        self.input_tables = [table_left, table_right]
        self.merge_cols_left = cols_left if type(cols_left) is list else [cols_left]
        self.merge_cols_right = cols_right if type(cols_right) is list else [cols_right]     
        self.output_filter = output_filter
        self.input_constraint = None
        self.rename_left = {k:k for k,v in self.input_tables[0][0].values.items()}
        self.rename_right = {k:k for k,v in self.input_tables[1][0].values.items()}
        for col_name_l,v in self.input_tables[0][0].values.items():
            for col_name_r,v in self.input_tables[1][0].values.items():
                if col_name_l == col_name_r and col_name_l not in self.merge_cols_left and col_name_r not in self.merge_cols_right:
                   self.rename_left[col_name_l] = '{}_x'.format(col_name_l)
                   self.rename_right[col_name_r] = '{}_y'.format(col_name_r)

    def check_small_model(self, check_superset=False):
        # 𝑓(𝑓𝑗𝑜𝑖𝑛(𝑡_𝑙,𝑡_𝑟 ))=𝐹𝑎𝑙𝑠𝑒 ⋀ 𝑓𝑚𝑎𝑡𝑐ℎ(𝑡_𝑙,𝑡_𝑟 )=𝑇𝑟𝑢𝑒 → 𝐹(𝐿𝐽(𝑡_𝑙,𝑇_𝑟 ))=∅
        # one of the right row is joinable and the output of join cannot pass f, then the other rows cannot be joined/does not pass filter
        table_left = self.input_tables[0]
        table_right = self.input_tables[1]
        temp_tup1 = {self.rename_left[k]:v for k,v in table_left[0].values.items()}
        temp_tup1.update({self.rename_right[k]:v for k,v in table_right[0].values.items()})
        temp_tup2 = {self.rename_left[k]:v for k,v in table_left[0].values.items()}
        temp_tup2.update({self.rename_right[k]:v for k,v in table_right[1].values.items()})
        f1 = (getv(zeval(self.output_filter, Tuple(temp_tup1)))==False)
        f2 = z3.And(*[table_left[0][self.merge_cols_left[i]] == table_right[0][self.merge_cols_right[i]] for i in range(len(self.merge_cols_left))])==True
        tup2_joinable = z3.And(*[table_left[0][self.merge_cols_left[i]] == table_right[1][self.merge_cols_right[i]] for i in range(len(self.merge_cols_left))])
        fimplies = z3.Implies(tup2_joinable, getv(zeval(self.output_filter, Tuple(temp_tup2)))==False) # either not joinable, or joinable but cannot pass f
        cond1 = z3.Implies(z3.And(f1, f2), fimplies)
        
        if check_superset:
            cond1=True
        # print("f1 = {}".format(z3.simplify(zeval(self.output_filter, Tuple(temp_tup1)))))
        # print("f2 = {}".format(z3.simplify(f2)))
        # print("cond1: {}".format(z3.simplify(cond1)))

        # 𝑓(𝑓𝑗𝑜𝑖𝑛(𝑡_𝑙,𝑡_𝑟 ))=𝑇𝑟𝑢𝑒 ⋀ 𝑓𝑚𝑎𝑡𝑐ℎ(𝑡_𝑙,𝑡_𝑟 )=𝑇𝑟𝑢𝑒 → 𝑔_2 (𝑡_𝑟 )=𝑇𝑟𝑢𝑒
        f1 = (getv(zeval(self.output_filter, Tuple(temp_tup1)))==True)
        fimplies = getv(zeval(self.input_filters[1], table_right[0]))==True
        cond2 = z3.Implies(z3.And(f1, f2), fimplies)

        if self.input_constraint is not None:
            return check_always_hold(z3.Implies(self.input_constraint, z3.And(cond1, cond2)))
        else:
            return check_always_hold(z3.And(cond1, cond2))

    def run_operator(self, input_tables):
        table_left = input_tables[0]
        table_right = input_tables[1]
        output = []
        right_values = set([k for k,v in table_right[0].values.items()])
        right_values = right_values - set(self.merge_cols_right)
        for left_t in table_left:
            join_conds = []
            for right_t in table_right:
                join_cond = z3.And(*[left_t[self.merge_cols_left[i]] == right_t[self.merge_cols_right[i]] for i in range(len(self.merge_cols_left))])
                join_conds.append(z3.And(right_t.eval_exist_cond(), join_cond))
                temp = {self.rename_left[k]:v for k,v in left_t.values.items()}
                temp.update( {self.rename_right[k]:right_t.values[k] for k in right_values})
                t = Tuple(temp, z3.And(left_t.eval_exist_cond(), right_t.eval_exist_cond(), join_cond))
                output.append(t)
            cannot_join = z3.And(*[xx==False for xx in join_conds])
            temp = {self.rename_left[k]:v for k,v in left_t.values.items()}
            temp.update( {self.rename_right[k]:Value(getv(table_right[0].values[k]), True) for k in right_values})
            t = Tuple(temp, z3.And(left_t.eval_exist_cond(), cannot_join))
            output.append(t)
        return output
    def lineage_exact(self):
        equiv_pairs = get_equivalent_pairs(self.input_filters[0])
        equiv_pairs = equiv_pairs + get_equivalent_pairs(self.input_filters[1])
        cols_match_value = [p1.get_full_name() if isinstance(p1, UnresolvedAttribute) and isinstance(p2, Literal) else None for p1,p2 in equiv_pairs]
        cols_match_value += [p2.get_full_name() if isinstance(p2, UnresolvedAttribute) and isinstance(p1, Literal) else None for p1,p2 in equiv_pairs]
        cols_match_value = set(list(filter(lambda x: x is not None, cols_match_value)))
        if all([is_col_in_exist_columns(c, cols_match_value) for c in self.merge_cols_left]):
            return [True, True]
        if all([is_col_in_exist_columns(c, cols_match_value) for c in self.merge_cols_right]):
            return [True, True]
        return [False, False]

class ProjectionInference(OperatorInference):
    def __init__(self, input_table, col_func_pair, output_filter): # col_func_pair: (new_column, transform_func, return_type) triple
        self.input_tables = [input_table]
        self.col_func_pair = col_func_pair
        self.output_filter = output_filter
        self.input_constraint = None
    def run_operator(self, input_tables):
        table = input_tables[0]
        ret = []
        # print("Run project: {}".format(self.input_filters[0]))
        # print("output filter = {}".format(self.output_filter))
        # print("projection list: {}".format(', '.join(['{}:{}'.format(k,v) for k,v in self.col_func_pair])))
        for tup in table:
            new_values = {}
            for new_col, transform_func in self.col_func_pair:
                new_col = new_col.get_column_name()
                if transform_func is None:
                    newv = tup[new_col]
                else:
                    newv = zeval(transform_func, tup)
                new_values[new_col] = newv
            ret.append(Tuple(new_values, tup.eval_exist_cond(), tup.count))
        return ret
    
class GroupByInference(OperatorInference):
    def __init__(self, input_table, grouping_expressions, aggregation_expressions, output_filter):
        self.input_tables = [input_table]
        self.grouping_expressions = grouping_expressions
        self.aggregation_expressions = aggregation_expressions
        self.groupby_used_cols = set.union(*[set(get_columns_used(expr)) for expr in self.grouping_expressions]) if len(self.grouping_expressions)>0 else set()
        self.output_columns_group = [get_new_column_name_from_expr(expr) for expr in grouping_expressions]
        self.output_columns_aggr = [get_new_column_name_from_expr(expr) for expr in aggregation_expressions]
        self.output_filter = output_filter
        self.input_constraint = None
    def run_operator(self, input_tables):
        table = input_tables[0]
        values = {}
        for j,expr in enumerate(self.aggregation_expressions):
            rs = compute_aggregate(expr, table)
            values[self.output_columns_aggr[j]] = rs
        # for i,expr in enumerate(self.grouping_expressions):
        #     values[self.output_columns_group[i]] = zeval(expr, table[0])
        exist_cond = z3.Or(*[tup.eval_exist_cond() for tup in table])
        t = create_tuple(values, exist_cond, 1)
        return [t]
    def check_small_model(self, check_superset=False):
        # prop 1: f(agg(T+U))==\empty <=> f(agg(T))==\empty and f(agg(U))==\empty
        # ------- if superset, prop 1: f(agg(T+U))==\empty & g(U)==\empty => f(agg(T))=\empty
        # prop 2: f(agg(T+U))!=\empty & g(U)==\empty => agg(T+U)==agg(T)
        # prop 3: f(agg(T))==\empty => g(T)==\empty
        if all([col in self.groupby_used_cols for col in get_columns_used(self.input_filters[0])]) and all([col in self.groupby_used_cols for col in get_columns_used(self.output_filter)]):
            return True
        if check_superset and all([col in self.groupby_used_cols for col in get_columns_used(self.input_filters[0])]):
            return True
        table = self.input_tables[0]
        same_group_assumption = True
        for expr in self.grouping_expressions:
            same_group_assumption = z3.And(same_group_assumption,\
                z3.And(*[getv(zeval(expr, table[0])) == getv(zeval(expr, table[i])) for i in range(1,len(table))])) # TODO, .equals instead of ==?

        t0_values = {}
        t1_values = {}
        t2_values = {}
        for j,expr in enumerate(self.aggregation_expressions):
            rs_0 = compute_aggregate(expr, table)
            rs_1 = compute_aggregate(expr, [table[1]])
            rs_2 = compute_aggregate(expr, [table[0]])
            #print("aggr {} : rs_0 = {}".format(expr_to_sql(expr), z3.simplify(getv(rs_0))))
            t0_values[self.output_columns_aggr[j]] = rs_0
            t1_values[self.output_columns_aggr[j]] = rs_1
            t2_values[self.output_columns_aggr[j]] = rs_2
        # for j,expr in enumerate(self.grouping_expressions):
        #     t0_values[self.output_columns_group[j]] = zeval(expr, table[0])
        #     t1_values[self.output_columns_group[j]] = zeval(expr, table[0])
        #     t2_values[self.output_columns_group[j]] = zeval(expr, table[0])
        t0 = Tuple(t0_values)
        t1 = Tuple(t1_values)
        t2 = Tuple(t2_values)
        pre_cond_1 = z3.Implies(getv(zeval(self.output_filter, t0))==False, getv(zeval(self.output_filter,t2))==False)
        
        pre_cond_2 = []
        for col_name, some_v in t0_values.items():
            pre_cond_2.append(z3.Implies(z3.And(getv(zeval(self.output_filter,t0))==True, getv(zeval(self.input_filters[0],table[0]))==False), \
                                t0_values[col_name] == t1_values[col_name] ))
        if check_superset:
            pre_cond_1 = z3.Implies(z3.And(getv(zeval(self.output_filter,t0))==False, getv(zeval(self.input_filters[0],table[0]))==False), \
                                getv(zeval(self.output_filter, t1))==False)
        #vs = self.get_all_table_variables()
        if self.input_constraint is not None:
            return check_always_hold(z3.Implies(z3.And(self.input_constraint,\
                                            same_group_assumption), z3.And(pre_cond_1, z3.And(*pre_cond_2))))
        else:
            return check_always_hold(z3.Implies(same_group_assumption, z3.And(pre_cond_1, z3.And(*pre_cond_2))))
    def verify_correct(self, check_superset=False):
        if all([col in self.groupby_used_cols for col in get_columns_used(self.input_filters[0])]) and all([col in self.groupby_used_cols for col in get_columns_used(self.output_filter)]):
            return True
        if check_superset and all([col in self.groupby_used_cols for col in get_columns_used(self.input_filters[0])]):
            return True
        rs1_temp = self.run_operator(self.input_tables)
        rs1 = self.run_output_filter(rs1_temp)
        #print("Run f(op())")
        rs1_temp = self.run_operator(self.input_tables)
        rs1 = self.run_output_filter(rs1_temp)
        #print("rs1 {}".format(z3_simplify(rs1[0].eval_exist_cond())))

        #print("")
        #print("Run op(g())")
        rs2_temp = self.run_input_filter(self.input_tables, self.input_filters)
        rs2 = self.run_operator(rs2_temp)
        if check_superset:
            rs2 = self.run_output_filter(rs2)
        #print("rs2 {}".format(z3_simplify(rs2[0].eval_exist_cond())))
        
        #print("input filter = {}".format(expr_to_sql(self.input_filters[0])))
        table = self.input_tables[0]
        same_group_assumption = True
        for expr in self.grouping_expressions:
            same_group_assumption = z3.And(same_group_assumption,\
                z3.And(*[getv(zeval(expr, table[0]))==getv(zeval(expr, table[i])) for i in range(1,len(table))])) # TODO, .equals instead of ==?
        if len(rs1) == len(rs2):
            if len(self.grouping_expressions) == 0:
                expr = z3.And(*[z3.And(*[v_ == (rs2[i][col]) for col,v_ in rs1[i].values.items()]) for i in range(len(rs1))])
            else:
                expr = z3.Implies(same_group_assumption, z3.And(*[z3.And(rs1[i].eval_exist_cond() == rs2[i].eval_exist_cond(),\
                                z3.Implies(rs1[i].eval_exist_cond(), z3.And(*[v_ == (rs2[i][col]) for col,v_ in rs1[i].values.items()]))) for i in range(len(rs1))]))
                
            vs = self.get_all_table_variables()
            if self.input_constraint is not None:
                return check_always_hold(z3.Implies(self.input_constraint, expr))
            else:
                return check_always_hold(expr)
        else:
            assert(False, "TODO: CANNOT VERIFY FOR DIFFERENT NUMBER OF TUPLES")

class PivotInference(GroupByInference):
    def __init__(self, input_table, group_columns, pivotColumn, pivotValues, aggregates, output_schema, output_filter):
        self.pivotColumn = pivotColumn
        self.pivotValues = pivotValues
        self.aggregates = aggregates
        self.group_columns = group_columns
        self.output_schema = output_schema
        new_aggrs = [c for c in group_columns]
        for v in pivotValues:
            actual_value = v.child
            for a in aggregates:
                if isinstance(a, Alias):
                    new_col_name = get_new_column_name_from_expr(v)+'_'+get_new_column_name_from_expr(a)
                else:
                    new_col_name = get_new_column_name_from_expr(v)
                if isinstance(a, Alias):
                    a = a.child
                if a.name.lower() in ['count']:
                    new_aggr = UnresolvedFunction('sum', \
                                                  [If(EqualTo(pivotColumn, actual_value), Literal(1,'integer'), Literal(0,'integer'))], False)
                elif a.name.lower() in ['sum']:
                    new_aggr = UnresolvedFunction('sum', \
                                                   [If(EqualTo(pivotColumn, actual_value), a.children[0], Literal(0,'integer'))], False)
                elif a.name.lower() in ['distinct']:
                    typ = self.output_schema[-1][1]
                    new_aggr = UnresolvedFunction('distinct', \
                                                   [If(EqualTo(pivotColumn, actual_value), a.children[0], Literal(get_init_value_by_type(typ), typ))], False)
                else:
                    print("Pivot with {} aggregation is not supported yet")
                    assert(False) 
                new_aggrs.append(Alias(new_aggr, new_col_name))
        # for a in new_aggrs:
        #     print("new aggregation {}".format(expr_to_sql(a))) 
        super().__init__(input_table, group_columns, new_aggrs, output_filter)

class NoChangeInference(OperatorInference):
    def __init__(self, input_tables, output_filter):
        self.input_tables = input_tables
        self.output_filter = output_filter
        self.input_constraint = None
    def run_operator(self, input_tables):
        table = input_tables[0]
        new_table = [Tuple(tup.values, tup.exist_cond, tup.count) for tup in table] 
        return new_table
    
# one row to multiple rows
class UnpivotInference(OperatorInference): 
    def __init__(self, input_table, id_vars, value_vars, var_name, value_name, output_filter):
        self.input_tables = [input_table]
        self.output_filter = output_filter
        self.id_vars = id_vars
        self.value_vars = value_vars
        self.var_name = var_name
        self.value_name = value_name
        self.input_constraint = None
    def run_operator(self, input_tables):
        ret = []
        for tup in input_tables[0]:
            for i,value_var in enumerate(self.value_vars):
                tup_values = {id_var:tup[id_var] for id_var in self.id_vars}
                tup_values[self.var_name] = Literal(value_var,'str')
                tup_values[self.value_name] = tup[value_var]
                ret.append(Tuple(tup_values, tup.eval_exist_cond()))
        return ret
    
class TopNInference(OperatorInference):
    def __init__(self, input_table, N, sort_order, desc, output_filter):
        self.input_tables = [input_table]
        self.N = N
        self.sort_order = sort_order
        self.desc = desc
        self.output_filter = output_filter
        self.input_constraint = None
    def check_small_model(self, check_superset=False):
        if check_superset:
            return True
        # descending order: 
        # |f(topN(T))| < N -> forall t <= min(T), g(t) = False
        assert(len(self.input_tables[0])>= self.N)
        table = self.input_tables[0]
        cur_count = 0
        for tup in self.input_tables[0][:self.N]:
            exist_cond = z3.And(zeval(self.output_filter, tup), cur_count<self.N)
            cur_count = z3.If(exist_cond, cur_count+1, cur_count)
        assump1 = z3.And(cur_count>0, cur_count < self.N)
        
        if self.desc:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v>=table[i+1][self.sort_order[0]].v)
            newtup_values = {}
            for k,v in table[0].values.items():
                newtup_values[k] = Value(get_symbolic_value_by_type(get_z3type_by_variable(v.v),'other-{}'.format(k)))#get_new_variable_by_type(get_variable_type(v.v),' other-{}'.format(k)))
            minv = table[0][self.sort_order[0]].v # TODO: sort order only 1 column for now
            for row in table[1:]:
                minv = z3.If(minv<row[self.sort_order[0]].v, minv, row[self.sort_order[0]].v)
            target = zeval(self.output_filter, Tuple(newtup_values))==False
            expr = z3.Implies(z3.And(descending_constr, newtup_values[self.sort_order[0]].v<=minv, assump1), target)
            #print("expr = {}".format(z3.simplify(target)))
        else:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v<=table[i+1][self.sort_order[0]].v)
            newtup_values = {}
            for k,v in table[0].values.items():
                newtup_values[k] = Value(get_symbolic_value_by_type(get_z3type_by_variable(v.v),'other-{}'.format(k)))
            maxv = table[0][self.sort_order[0]].v # TODO: sort order only 1 column for now
            for row in table[1:]:
                maxv = z3.If(maxv>row[self.sort_order[0]].v, maxv, row[self.sort_order[0]].v)
            target = zeval(self.output_filter, Tuple(newtup_values))==False
            expr = z3.Implies(z3.And(descending_constr, newtup_values[self.sort_order[0]].v>=maxv, assump1), target)
            #print("expr = {}".format(z3.simplify(target)))

        vs = self.get_all_table_variables()+[newtup_values[self.sort_order[0]].v]
        if self.input_constraint is not None:
            return check_always_hold(z3.Implies(self.input_constraint, expr))
        else:
            return check_always_hold(expr)

    def run_operator(self, input_tables):
        new_table = []
        cur_count = 0
        for tup in input_tables[0]:
            new_tup = {k:v for k,v in tup.values.items()}
            exist_cond = z3.And(tup.eval_exist_cond(), cur_count<self.N)
            cur_count = z3.If(exist_cond, cur_count+1, cur_count)
            new_table.append(Tuple(new_tup, exist_cond))
        return new_table
    
    def verify_correct(self, check_superset=False):
        table = self.input_tables[0]
        if self.desc:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v>=table[i+1][self.sort_order[0]].v)
        else:
            descending_constr = True
            for i in range(len(table)-1):
                descending_constr = z3.And(descending_constr, table[i][self.sort_order[0]].v<=table[i+1][self.sort_order[0]].v)
        
        if self.input_constraint is not None:
            self.input_constraint = z3.And(self.input_constraint, descending_constr)
        else:
            self.input_constraint = descending_constr
        return super().verify_correct(check_superset)
    

class SubqueryInference(OperatorInference):
    def __init__(self, input_table, condition, subquery_cmp, subppl):
        self.input_tables = [input_table]
        self.condition = condition # rest of query condition
        self.subquery_cmp = subquery_cmp
        self.subppl = subppl # a list of PipelineNode
        self.input_constraint = None
        self.correlated_columns = []
    # assume input_tables[0] is outer table, input_tables[1] is inner table
    def check_small_model(self, check_superset=False):
        return True
    def run_operator(self, input_tables):
        output = []
        for t in input_tables[0]:
            output_t = Tuple(t.values, getv(zeval(And(t.exist_cond, self.condition), t)), t.count)
            output.append(output_t)
        return output
    def lineage_exact(self):
        #print("Subquery correlated columns = {}".format(self.correlated_columns))
        column_used = get_columns_used(self.output_filter)
        #equiv_pairs = get_equivalent_pairs
        if not all([is_col_in_exist_columns(c, column_used) for c in self.correlated_columns]):
            return [True, False]
        return [True, True]
    # FIXME: remove
    #def verify_pushup_correct(self)
    

"""
class SubpipeInputInference(OperatorInference):
    def __init__(self, input_table, input_type, group_key, output_filter):
        self.input_type = input_type
        self.group_key = group_key
        self.output_filter = output_filter
        self.input_tables = [input_table]
    def run_operator(self, input_tables):
        return self.input_tables[0] 

class CrosstableUDFOnePathInference(OperatorInference):
    def __init__(self, input_table, new_col, sub_pipeline, sub_pipeline_dependency, path_cond, output_filter):
        self.input_tables = [input_table]
        self.new_col = new_col
        self.sub_pipeline = sub_pipeline
        self.sub_pipeline_dependency = sub_pipeline_dependency # key: sub_pipeline_idx, value: [idxes]
        self.path_cond = path_cond
        self.input_constraint = None 
        self.output_filter = output_filter
        self.input_filters = []
        # row + table -> one_value
        # last is SetItem...

        #for path in self.sub_pipeline:

    def run_operator(self, input_tables):
        ret = []
        #for i,op in enumerate(self.sub_pipeline):
        #    print("****** subpipe op = {}".format(op))
        for tup in input_tables[0]:
            out_value_map = {} 
            for i,op in enumerate(self.sub_pipeline):
                print(type(op))
                #print("OPERATOR {} input filters : {}".format(type(op),','.join([str(x) for x in op.input_filters])))

                inputs = []
                if isinstance(op, SubpipeInputInference):
                    if op.input_type == 'row':
                        out_value_map[i] = tup
                    else:
                        out_value_map[i] = input_tables[1]
                else:
                    for dep_id in self.sub_pipeline_dependency[i]:
                        inputs.append(out_value_map[dep_id])
                    output = op.run_operator(inputs)
                    out_value_map[i] = output
                    #print("OUTPUT for {} is {}".format(type(op), output[0] if type(output) is list else output))
            values = {k:v for k,v in tup.values.items()}
            last_op_idx = len(self.sub_pipeline)-1 if type(self.path_cond) is bool else len(self.sub_pipeline)-2
            final_out = out_value_map[last_op_idx]
            final_exist = True
            if type(final_out) is list:
                final_out_v = [v for k,v in final_out[0].values.items()][0].v
                final_exist = final_out[0].eval_exist_cond()
            elif type(final_out) is Tuple:
                final_out_v = final_out.values['scalar_out'].v
                final_exist = final_out.eval_exist_cond()
            values[self.new_col] = Value(final_out_v)
            #print("--------FINAL RETURN VALUE = {}".format(z3_simplify(final_out_v)))
            #print('--------final exist = {}'.format(z3_simplify(final_exist)))
            #ret.append(Tuple(values, z3.And(final_exist, tup.eval_exist_cond())))
            ret.append(Tuple(values, tup.eval_exist_cond()))

        if not type(self.path_cond) is bool:
            path_cond = out_value_map[len(self.sub_pipeline)-1].values['scalar_out'].v
        else:
            path_cond = True
        return ret, path_cond
    
    def output_exists(self):
        table_right_op = None
        for op in self.sub_pipeline:
            #if len(self.input_filters) == 0 and isinstance(op, SubpipeInputInference) and op.input_type == 'row' and len(op.input_filters) > 0:
            #    self.input_filters = op.input_filters
            if isinstance(op, SubpipeInputInference) and op.input_type == 'table':
                table_right_op = op
        
        final_output,path_cond1 = self.run_operator(self.input_tables+table_right_op.input_tables)
        self.input_constraint = zeval(self.output_filter, final_output[0])==True
        return self.input_constraint

    def verify_correct(self, check_superset=False):
        table_right_op = None
        for op in self.sub_pipeline:
            #if len(self.input_filters) == 0 and isinstance(op, SubpipeInputInference) and op.input_type == 'row' and len(op.input_filters) > 0:
            #    self.input_filters = op.input_filters
            if isinstance(op, SubpipeInputInference) and op.input_type == 'table':
                table_right_op = op
        #print("INPUT FILTER ::::: {}".format(len(self.input_filters)))
        #for i in self.input_filters:
        #    print(i)
        #if len(self.input_filters) == 0:
        #    return True
        left_eval = zeval(self.input_filters[0], self.input_tables[0][0])
        final_output,path_cond1 = self.run_operator(self.input_tables+table_right_op.input_tables)
        final_output_exist = zeval(self.output_filter, final_output[0])
        #print("final output 1 = {}, 2 = {}".format(self.output_filter, final_output[0].exist_cond))
        #print("**** {}".format(z3_simplify(final_output[0]['isin_country'].v)))
        #print("---- {}".format(final_output_exist))
        #print("FINAL OUTPUT EXISTS = {}".format(z3_simplify(final_output_exist)))

        # ∀𝑡_𝑙,𝑡_𝑟, 𝑔_l (𝑡_𝑙)=𝐹𝑎𝑙𝑠𝑒 → 𝑓(𝑂𝑝(𝑡_𝑙,𝑡_𝑟 ))=𝐹𝑎𝑙𝑠𝑒
        assumption1 = z3.Implies(left_eval==False, final_output_exist==False)

        # ∀𝑡_𝑙, 𝑇_𝑟,𝑔_l (𝑡_𝑙)=𝑇𝑟𝑢𝑒 → 𝑓(𝑂𝑝(𝑡_𝑙,𝑇_𝑟 ))=𝑂𝑝(𝑡_𝑙, 𝑔_r (𝑇_𝑟 )) 
        assumption2 = []
        #rs2_temp = self.run_input_filter(self.input_tables, self.input_filters)
        #rs2_temp_right = table_right_op.run_input_filter(table_right_op.input_tables, table_right_op.input_filters)
        rs2_temp = self.run_input_filter(self.input_tables+table_right_op.input_tables, self.input_filters)
        output_with_input_filter,path_cond2 = self.run_operator(rs2_temp)
        if check_superset:
            output_with_input_filter = self.run_output_filter(output_with_input_filter)
        # path condition same
        assumption0 = z3.Implies(left_eval==True, (path_cond1==path_cond2))
        
        
        for i in range(len(final_output)):
            #print("FINAL OUTPUT EXIST = {}".format(z3.simplify(final_output_exist)))
            #print("input filter, exist = {}".format(z3.simplify(output_with_input_filter[i].eval_exist_cond())))

            #print("**** FINAL rs1 = {} ".format(z3_simplify(final_output[0].values[self.new_col].v)))
            #print("**** FINAL rs2 = {} ".format(z3_simplify(output_with_input_filter[0].values[self.new_col].v)))
            #assumption2.append(z3.And(final_output_exist == output_with_input_filter[i].eval_exist_cond()))
            #assumption2.append(z3.Implies(final_output_exist==True, final_output[i].values[self.new_col].v==output_with_input_filter[i].values[self.new_col].v))
            assumption2.append(z3.And(final_output_exist == output_with_input_filter[i].eval_exist_cond(),\
            z3.Implies(final_output_exist==True, final_output[i].values[self.new_col].v==output_with_input_filter[i].values[self.new_col].v)))
        assumption2 = z3.Implies(left_eval==True, z3.And(*assumption2))

        vs = self.get_all_table_variables()
        vs = vs + table_right_op.get_all_table_variables()
        # print("path cond 1 = {}".format(z3_simplify(path_cond1)))
        # print("path cond 2 = {}".format(z3_simplify(path_cond2)))
        # print("input constraint = {}".format(z3_simplify(self.input_constraint)))
        if self.input_constraint is not None:
            self.input_constraint = z3.And(self.input_constraint, path_cond1==True)
        else:
            self.input_constraint = (path_cond1==True)

        return check_always_hold(z3.Implies(self.input_constraint, z3.And(assumption0, assumption1, assumption2)), vs)
        

class CrosstableUDFInference(OperatorInference):
    def __init__(self, inferences, output_filter):
        self.inferences = inferences
        self.output_filter = output_filter
    def verify_correct(self, check_superset=False):
        x = []
        for i in self.inferences:
            x.append(i.verify_correct(check_superset))
        return all(x)
    
"""
