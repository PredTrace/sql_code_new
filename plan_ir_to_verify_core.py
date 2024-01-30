import os
import sys
import z3
from verify_core_op import *
from ir import *
from util import *
from z3_eval_plan_ir import *


def walk_plan_tree(node, is_stop_node=lambda x:False, handle_rs=lambda x:x, union_rs=lambda x,y:x): # handle_rs/union_rs are lambdas
    if is_stop_node(node):
        return handle_rs(node)
    if hasattr(node, 'child') and node.child is not None:
        ret = walk_plan_tree(node.child, is_stop_node, handle_rs, union_rs)
        return union_rs(handle_rs(node), ret)
    else:
        assert(False)
    
def get_lineage_filter_from_query(plan_node, table_schema):
    ret = walk_plan_tree(plan_node, is_stop_node=lambda x:isinstance(x,Filter),\
                   handle_rs=lambda x: x.condition if isinstance(x,Filter) else None,\
                   union_rs=lambda x,y: x if y is None else y)
    resolve_literal_type(ret, table_schema)
    if ret is None:
        return True
    return ret

def infer_schema(ppl_node, table_schemas):
    plan_node = ppl_node.plan_node
    table_column_type_map = []
    for up in ppl_node.upstream_nodes:
        column_types = {k:v for k,v in up.output_schema}
        column_types.update({k.split('.')[-1]:v for k,v in up.output_schema})
        table_column_type_map.append(column_types)

    if isinstance(plan_node, UnresolvedRelation):
        ppl_node.input_schema = [table_schemas[plan_node.tableIdentifier]]
        ppl_node.input_schema = [[('{}.{}'.format(plan_node.tableIdentifier,k), v) for k,v in ppl_node.input_schema[0]]]
        ppl_node.output_schema = ppl_node.input_schema[0]
        
    elif isinstance(plan_node, Filter) or \
        isinstance(plan_node, LocalLimit) or \
        isinstance(plan_node, LocalLimit) or \
        isinstance(plan_node, Intersect) or \
        isinstance(plan_node, Union) or \
        isinstance(plan_node, Sort) or \
        isinstance(plan_node, Distinct):
        
        ppl_node.input_schema = [ppl_node.upstream_nodes[0].output_schema]
        ppl_node.output_schema = ppl_node.input_schema[0]
    
    elif isinstance(plan_node, Project):
        ppl_node.input_schema = [ppl_node.upstream_nodes[0].output_schema]
        if any([isinstance(p, UnresolvedStar) for p in plan_node.projectList]):
            ppl_node.output_schema = ppl_node.input_schema[0]
        else:
            output_schema = []
            for p in plan_node.projectList:
                if isinstance(p, UnresolvedAttribute):
                    l1 = list(filter(lambda x: is_col_in_exist_columns(p.get_full_name(), [x[0]]), ppl_node.upstream_nodes[0].output_schema))
                    #l2 = list(filter(lambda x: x[0].split('.')[-1]==p.get_column_name(), ppl_node.upstream_nodes[0].output_schema))
                    #output_schema.append(l1[0] if len(l1) > 0 else l2[0])
                    output_schema.append(l1[0])
                elif isinstance(p, Alias):
                    # column_types = {}
                    # column_types.update({})
                    output_schema.append((p.name, infer_type(p.child, table_column_type_map[0])))
                elif projection_has_aggregation(p):
                    # FIXME
                    output_schema.append(('aggr_column', 'int'))#infer_type(p.children[0], table_column_type_map[0])))
                else:
                    print("{} not handled in infer schema, projection".format(p))
                    assert(False)
            ppl_node.output_schema = output_schema

    elif isinstance(plan_node, Join):
        ppl_node.input_schema = [ppl_node.upstream_nodes[0].output_schema, ppl_node.upstream_nodes[1].output_schema]
        ppl_node.output_schema = ppl_node.upstream_nodes[0].output_schema + ppl_node.upstream_nodes[1].output_schema
    
    elif isinstance(plan_node, SubqueryAlias):
        ppl_node.input_schema = [ppl_node.upstream_nodes[0].output_schema] 
        ppl_node.output_schema = [('{}.{}'.format(plan_node.alias, k.split('.')[-1]) if plan_node.alias is not None else k.split('.')[-1],v) \
                                for k,v in ppl_node.upstream_nodes[0].output_schema]
    
    elif isinstance(plan_node, Aggregate):
        ppl_node.input_schema = [ppl_node.upstream_nodes[0].output_schema]
        group_cols = [(get_columns_used(expr)[0], infer_type(expr, table_column_type_map[0])) for expr in plan_node.groupingExpressions]
        aggr_cols = [(get_new_column_name_from_expr(expr), infer_type(expr, table_column_type_map[0])) for expr in plan_node.aggregateExpressions]
        ppl_node.output_schema = aggr_cols #group_cols + aggr_cols

    elif isinstance(plan_node, Sort):
        ppl_node.input_schema = [ppl_node.upstream_nodes[0].output_schema]
        ppl_node.output_schema = ppl_node.input_schema[0]
    
    elif isinstance(plan_node, Pivot):
        ppl_node.input_schema = [ppl_node.upstream_nodes[0].output_schema]
        aggr_cols = []
        for a in plan_node.aggregates:
            aggr_cols.extend(get_columns_used(a))
        remaining_columns = list(filter(lambda c: c[0].split('.')[-1]!=plan_node.pivotColumn.get_column_name() \
                                        and not is_col_in_exist_columns(c[0], aggr_cols),\
                                       ppl_node.input_schema[0]))
        pivoted_columns = []
        for v in plan_node.pivotValues:
            for a in plan_node.aggregates:
                if isinstance(a, Alias):
                    new_name = get_new_column_name_from_expr(v)+'_'+get_new_column_name_from_expr(a)
                    a = a.child
                else:
                    new_name = get_new_column_name_from_expr(v)
                if isinstance(a, UnresolvedFunction) and a.name.lower() in builtin_aggr_functions and a.name.lower() not in ['distinct']:
                    new_type = 'int'
                else:
                    new_type = infer_type(a, table_column_type_map[0])
                pivoted_columns.append((new_name, new_type))
        ppl_node.output_schema = remaining_columns+pivoted_columns
    elif isinstance(plan_node, Unpivot):
        ppl_node.input_schema = [ppl_node.upstream_nodes[0].output_schema]
        unpivoted = [get_new_column_name_from_expr(e) for e in plan_node.values]
        value_typ = list(filter(lambda x: x[0].split('.')[-1]==unpivoted[0], ppl_node.input_schema[0]))[0][1]
        unpivoted_related_columns = [(n,value_typ) for n in plan_node.valueColumnNames] + [(plan_node.variableColumnName, 'str')]
        other_columns = list(filter(lambda x: x[0] not in unpivoted, ppl_node.input_schema[0]))
        ppl_node.output_schema = other_columns + unpivoted_related_columns

    else:
        print("infer schema unhandled: {}".format(type(plan_node)))
        assert(False)

def obtain_verify_core_op(plan_node, ppl_node, table_schemas):
    input_tables = [generate_symbolic_table('t{}'.format(i), ppl_node.input_schema[i], 1) for i in range(len(ppl_node.input_schema))]
    if isinstance(plan_node, Filter) and expr_has_subquery(plan_node.condition)==False:
        op = FilterInference(input_tables[0], plan_node.condition, None)
    elif (isinstance(plan_node, Filter) and expr_has_subquery(plan_node.condition)):
        subquery_node = get_subquery(plan_node.condition)[0] # Exists/InSubquery
        subquery_cmp = get_subquery_compare(plan_node.condition)[0]
        subppl = convert_plan_to_pipeline(subquery_node.query, table_schemas)
        op = SubqueryInference(input_tables[0], plan_node.condition, subquery_cmp, subppl)
        outertable_fields = set([k for k,v in input_tables[0][0].values.items()])
        outertable_fields = outertable_fields.union(set([k.split('.')[-1] for k,v in input_tables[0][0].values.items()]))
        def correlated_columns(ppl_node):
            if isinstance(ppl_node.inference, FilterInference):
                cur_schema = [col_name for col_name,col_type in ppl_node.output_schema]
                ret = list(filter(lambda x:x in outertable_fields and not (is_col_in_exist_columns(x, cur_schema)), get_columns_used(ppl_node.inference.condition)))
                return ret
            return []
        op.correlated_columns = subppl.iterate_pipeline_node(correlated_columns, union_rs = lambda x,y: x + y)
        if isinstance(op.subquery_cmp, InSubquery) or isinstance(op, EqualTo):
            field = op.subquery_cmp.field if isinstance(op.subquery_cmp, InSubquery) else op.subquery_cmp.left
            op.correlated_columns.append(field.get_full_name())
        #print("Correlated columns = {}".format(op.correlated_columns))

    elif isinstance(plan_node, Project):
        if any([projection_has_aggregation(p) for p in plan_node.projectList]):
            input_tables = [generate_symbolic_table('t{}'.format(i), ppl_node.input_schema[i], 2) for i in range(len(ppl_node.input_schema))]
            op = GroupByInference(input_tables[0], [], plan_node.projectList, None)

        elif any([isinstance(p, UnresolvedStar) for p in plan_node.projectList]):
            op = NoChangeInference([input_tables[0]], None)
        else:
            table_name = ppl_node.input_schema[0][0][0].split('.')[0]
            col_func_pair = [(UnresolvedAttribute([table_name, get_new_column_name_from_expr(p)]), \
                              p.child if isinstance(p, Alias) else p) for p in plan_node.projectList]
            op = ProjectionInference(input_tables[0], col_func_pair, None)
    elif isinstance(plan_node, SubqueryAlias):
        if plan_node.alias is not None:
            col_func_pair = [(UnresolvedAttribute([plan_node.alias, col_name.split('.')[-1]]), wrap_column_name(col_name)) for col_name,col_type in ppl_node.input_schema[0]]
            op = ProjectionInference(input_tables[0], col_func_pair, None)
        else:
            op = NoChangeInference([input_tables[0]], None)
    elif isinstance(plan_node, UnresolvedRelation):
        op = ReadTableInference(input_tables[0], None)
    elif isinstance(plan_node, Join):
        if plan_node.condition is None:
            join_left,join_right = [],[]
        else:
            assert(isinstance(plan_node.condition, EqualTo) and isinstance(plan_node.condition.left, UnresolvedAttribute) and isinstance(plan_node.condition.right, UnresolvedAttribute))
            
            schema_left = [col_name for col_name,col_type in ppl_node.upstream_nodes[0].output_schema]
            schema_right = [col_name for col_name,col_type in ppl_node.upstream_nodes[1].output_schema]
            if not is_col_in_exist_columns(plan_node.condition.left.get_full_name(), schema_left):
                join_right= plan_node.condition.left.get_full_name()
                join_left = plan_node.condition.right.get_full_name()
            else:
                join_left= plan_node.condition.left.get_full_name()
                join_right = plan_node.condition.right.get_full_name()

        if 'Inner' in plan_node.joinType:
            op = InnerJoinInference(input_tables[0], input_tables[1], join_left, join_right, None)
        else:
            input_tables = [generate_symbolic_table('t{}'.format(i), ppl_node.input_schema[i], 2) for i in range(len(ppl_node.input_schema))]
            op = LeftOuterJoinInference(input_tables[0], input_tables[1], join_left, join_right, None)
    elif isinstance(plan_node, Aggregate):
        # symbolic table of sz 2
        input_tables = [generate_symbolic_table('t{}'.format(i), ppl_node.input_schema[i], 2) for i in range(len(ppl_node.input_schema))]
        op = GroupByInference(input_tables[0], plan_node.groupingExpressions, plan_node.aggregateExpressions, None)
    elif isinstance(plan_node, Pivot):
        input_tables = [generate_symbolic_table('t{}'.format(i), ppl_node.input_schema[i], 2) for i in range(len(ppl_node.input_schema))]
        aggr_cols = []
        for a in plan_node.aggregates:
            aggr_cols.extend(get_columns_used(a))
        group_columns = list(filter(lambda c: c[0].split('.')[-1]!=plan_node.pivotColumn.get_column_name() \
                                        and not is_col_in_exist_columns(c[0], aggr_cols),\
                                       ppl_node.input_schema[0]))
        
        group_columns = [wrap_column_name(c[0]) for c in group_columns]
        op = PivotInference(input_tables[0], group_columns, plan_node.pivotColumn, plan_node.pivotValues, plan_node.aggregates, ppl_node.output_schema, None)

    elif isinstance(plan_node, Unpivot):
        value_vars = [get_new_column_name_from_expr(e) for e in plan_node.values]
        id_vars = [x[0].split('.')[-1] for x in filter(lambda x: x[0] not in value_vars, ppl_node.input_schema[0])]
        var_name = plan_node.variableColumnName
        value_name = plan_node.valueColumnNames[0]
        op = UnpivotInference(input_tables[0], id_vars, value_vars, var_name, value_name, None)

    elif isinstance(plan_node, Sort):
        op = NoChangeInference(input_tables, None)
    else:
        print("Unhandled: {}".format(plan_node))
        assert(False)
    
    ppl_node.inference = op
    return ppl_node


# make datetime string into int
from dateutil import parser
def resolve_literal_type(condition, table_schema):
    schema = {k:v for k,v in table_schema}
    schema.update({k.split('.')[-1]:v for k,v in table_schema})
    def update_x(x, schema):
        if isinstance(x, BinaryComparison) and isinstance(x.left, UnresolvedAttribute) and isinstance(x.right, Literal):
            typ = schema[x.left.get_full_name()] if x.left.get_full_name() in schema else schema[x.left.get_column_name()]
            if type_match(typ, x.right.dataType) == False:
                # TODO: more comprehensive literal convertion
                newv = parser.parse(str(x.right.value)).timestamp()
                datevalue_mapping.add_date(x.right.value, newv)
                x.right.value = newv
                x.right.dataType = typ
    walk_expr_tree(condition, is_stop_node=lambda x:isinstance(x, BinaryComparison), 
                   handle_rs=lambda x: update_x(x, schema),
                   union_rs=lambda x,y:x)

# plan traverses in the opposite direction of dataflow
# setup ppl graph first, then infer schema
def convert_plan_to_pipeline(plan_node, table_schemas):
    # some pre-steps to reorganzie the query plan
    # split filter with multiple subqueries into multiple filters
    if isinstance(plan_node, Filter):
        subq = get_subquery_compare(plan_node.condition) 
        if len(subq) > 1:
            new_cond = get_filter_replacing_compare(plan_node.condition, {subq[i]:True for i in range(1, len(subq))})
            cur_plan_node = plan_node
            cur_plan_node.conditoin = new_cond
            #print("new cond = {}".format(new_cond))
            next_op = plan_node.child
            for i in range(1, len(subq)):
                new_plan_node = Filter(subq[i], plan_node)
                #print("subq = {}".format(subq[i]))
                cur_plan_node.child = new_plan_node
                cur_plan_node = new_plan_node
            cur_plan_node.child = next_op

    # convert having clause into Aggregate-Filter-Projection
    if isinstance(plan_node, UnresolvedHaving) and projection_has_aggregation(plan_node.condition):
        aggrs = collect_aggregation(plan_node.condition)
        function_to_replace = {aggr:UnresolvedAttribute(['aggr_{}'.format(i)]) for i,aggr in enumerate(aggrs)}
        original_aggr = plan_node.child.aggregateExpressions
        plan_node.child.aggregateExpressions.extend([Alias(aggr, 'aggr_{}'.format(i)) for i,aggr in enumerate(aggrs)])
        condition = get_filter_replacing_function(plan_node.condition, function_to_replace)
        filter_node = Filter(condition, plan_node.child)
        # FIXME: Add proper projection, but now we do not know the schema
        # now I'm only looking at cases like 
        # select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300
        # where the query does not have an explicit projection
        projection_node = Project([wrap_column_name(get_new_column_name_from_expr(aggr)) for aggr in original_aggr], filter_node)
        plan_node = projection_node
        # print("projection: {}".format(','.join([str(s) for s in projection_node.projectList])))
        # print("filter node = {}".format(filter_node))
        # print("aggr node = {}".format(filter_node.child))

    ppl_node = PipelineNode(plan_node)
    if hasattr(plan_node, 'child') and plan_node.child is not None:
        child = convert_plan_to_pipeline(plan_node.child, table_schemas)
        ppl_node.add_upstream_node(child)
    elif isinstance(plan_node, Join):
        left = convert_plan_to_pipeline(plan_node.left, table_schemas)
        right = convert_plan_to_pipeline(plan_node.right, table_schemas)
        ppl_node.add_upstream_node(left)
        ppl_node.add_upstream_node(right)
    infer_schema(ppl_node, table_schemas)
    #print(plan_node)
    
    if isinstance(plan_node, Filter):
        #print(" + + + + + plan node condition = {} / {}".format(plan_node.condition, type(plan_node.condition)))
        resolve_literal_type(plan_node.condition, ppl_node.output_schema)
    obtain_verify_core_op(plan_node, ppl_node, table_schemas)
    return ppl_node


class DummyPplNode(object):
    def __init__(self, plan_node):
        self.plan_node = plan_node
        self.upstream_nodes = []
        self.downstream_nodes = []
        self.input_schema = []
        self.output_schema = []
    def add_upstream_node(self, ppl_node):
        self.upstream_nodes.append(ppl_node)
        ppl_node.downstream_nodes.append(self)

def infer_schema_only(plan_node, table_schemas):
    ppl_node = DummyPplNode(plan_node)
    if hasattr(plan_node, 'child') and plan_node.child is not None:
        child = infer_schema_only(plan_node.child, table_schemas)
        ppl_node.add_upstream_node(child)
    if isinstance(plan_node, Join):
        left = infer_schema_only(plan_node.left, table_schemas)
        right = infer_schema_only(plan_node.right, table_schemas)
        ppl_node.add_upstream_node(left)
        ppl_node.add_upstream_node(right)
    infer_schema(ppl_node, table_schemas)
    return ppl_node