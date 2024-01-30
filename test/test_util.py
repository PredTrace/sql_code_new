import os
import sys
sys.path.append('../')
from verify_core_op import *
from plan_ir_to_verify_core import *
from get_spark_logical_plan import *
from predicate_pushdown import *
from lineage_without_intermediate_result2 import *

def get_row_lineage(table_schemas, sql, lineage_filter_query, no_intermediate_result=False):
    plan = get_spark_logical_plan(sql)
    ppl = convert_plan_to_pipeline(plan, table_schemas)

    lineage_filter = get_lineage_filter_from_query(get_spark_logical_plan(lineage_filter_query), ppl.output_schema)
    if no_intermediate_result:
        ppl_nodes = lineage_inference_no_intermediate_result(ppl, lineage_filter)
    else:
        ppl_nodes = predicate_pushdown_pipeline(ppl, lineage_filter)
    print_filters_on_input_table(ppl_nodes)
    return ppl_nodes

# queries: [(sql, table_name)] pair
def get_lineage_from_queries(table_schemas, queries, lineage_filter_query, no_intermediate_result=False):
    plans = []
    output_schema = []
    for sql,table_name in queries:
        plan = get_spark_logical_plan(sql)
        plans.append(plan)
        output_schema = infer_schema_only(plan, table_schemas).output_schema
        output_schema = [(col_name.split('.')[-1], col_type) for col_name,col_type in output_schema]
        #table_schemas[table_name] = output_schema
        table_schemas.update({table_name:output_schema})
        #print("Set schema for {}: {}".format(table_name, output_schema))

    lineage_filter = get_lineage_filter_from_query(get_spark_logical_plan(lineage_filter_query), output_schema)
    lineage_filter_by_table = {queries[-1][1]:lineage_filter}
    for sql,table_name in reversed(queries):
        lineage_filter = lineage_filter_by_table[table_name]
        plan = plans.pop(-1)
        ppl = convert_plan_to_pipeline(plan, table_schemas)
        #ppl_nodes = predicate_pushdown_pipeline(ppl, lineage_filter)
        if no_intermediate_result:
            ppl_nodes = lineage_inference_no_intermediate_result(ppl, lineage_filter)
        else:
            ppl_nodes = predicate_pushdown_pipeline(ppl, lineage_filter)
        print_filters_on_input_table(ppl_nodes)

        for n in ppl_nodes:
            if isinstance(n.inference, ReadTableInference):
                lineage_filter_by_table[n.plan_node.tableIdentifier] = n.inference.input_filters[0]

    return lineage_filter_by_table



