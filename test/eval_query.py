import db_connection
import psycopg2

pgconn = db_connection.pgconn

def psql_run_query(sql, include_schema=False):
    cursor = pgconn.cursor()
    cursor.execute(sql)
    rs = cursor.fetchall()
    if include_schema:
        schema = [desc[0] for desc in cursor.description]
        return rs, schema
    else:
        return rs
    # rs = []
    # row = cursor.fetchone()
    # while row is not None:
    #     rs.append(row)
    #     row = cursor.fetchone()
    # #cursor.commit()
    # return rs


import sys
sys.path.append('../')
from verify_core_op import ReadTableInference, SubqueryInference
from z3_eval_plan_ir import expr_to_sql, simplify_pred, replace_rowsel_variables, get_equivalent_pairs, AllAnd
from ir import EqualTo

sys.path.append('./tpch/')
from tpch_schema import TPCH_SCHEMA

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
        rowsel_variable_values[rowsel_variable] = [row[i] for row in rs]
    return rowsel_variable_values

def clean_sql(sql):
    return sql.replace("extract('year',l_shipdate)", "extract(year from l_shipdate)").replace("extract('year',o_orderdate)", 'extract(year from o_orderdate)')

def process_intermidate_rs(sql, lineage_pred):
    p_g = sql.find('group by')
    if p_g == -1:
        p_g = sql.find('order by')
    p_f = sql.rfind('from',)
    new_sql = "select * " + sql[p_f:p_g] + ' and {}'.format(lineage_pred)
    new_sql = clean_sql(new_sql)
    print(new_sql)
    return psql_run_query(new_sql, include_schema=True)

import time
def run_lineage_query(orig_sql, ppl_nodes, reference_data={}, subquery_level=0):
    indent = ''.join(['\t' for i in range(subquery_level)])
    if subquery_level == 0:
        print("\n========================")
        print("Final pushed filters on the input tables:\n")
    else:
        print("{}----subquery:-----\n".format(indent))
    readtable_nodes = []
    for n in ppl_nodes:
        if isinstance(n.inference, ReadTableInference):
            readtable_nodes.append(n)
    if hasattr(n, 'processing_order'):
        readtable_nodes.sort(key=lambda x:x.processing_order)
    
    rowsel_variable_values = {k:v for k,v in reference_data.items()}

    save_intermeidate = any([ hasattr(n, 'intermediate_result_row_sel_pred') for n in ppl_nodes])
    if save_intermeidate:
        
        intermediate_pred = AllAnd(*[n.intermediate_result_row_sel_pred for n in list(filter(lambda n: hasattr(n, 'intermediate_result_row_sel_pred'), ppl_nodes))])
        lineage_pred = expr_to_sql(list(filter(lambda n: hasattr(n, 'intermediate_result_row_sel_pred'), ppl_nodes))[-1].original_predicate)
        intermediate_rs, intermediate_schema = process_intermidate_rs(orig_sql, lineage_pred)
        new_rowsel = convert_rs_to_rowsel_variable(intermediate_rs, intermediate_schema, intermediate_pred)
        # for k,v in new_rowsel.items():
        #     print("{}\treplacing {} with {}".format(indent, k, str(v) if len(v) < 5 else '{}...'.format(v[:5])))
        rowsel_variable_values.update(new_rowsel)

    total_time = 0

    ret = {} # key: readtable_nodes, value: #rows
    for n in readtable_nodes:
        lineage_pred = simplify_pred(n.inference.input_filters[0])
        lineage_pred = expr_to_sql(replace_rowsel_variables(lineage_pred, rowsel_variable_values))
        keys = ','.join([col_name for col_name,col_type in TPCH_SCHEMA[n.plan_node.tableIdentifier] if 'key' in col_name])
        #lineage_sql = "SELECT * FROM {} WHERE {}".format(n.plan_node.tableIdentifier, lineage_pred)
        lineage_sql = "SELECT {} FROM {} WHERE {}".format(keys, n.plan_node.tableIdentifier, lineage_pred)
        lineage_sql = clean_sql(lineage_sql)
        start = time.time()
        rs, schema = psql_run_query(lineage_sql, include_schema=True)
        elapse = time.time()-start
        print("run for {} sec".format(elapse))
        total_time += elapse
        ret[n] = len(rs)
        print("{}query on table {} : \n{}----{}, return {} rows\n".format(indent, n.plan_node.tableIdentifier, indent, lineage_pred[:100], len(rs)))
        if hasattr(n, 'row_sel_pred_derived') and n.row_sel_pred_derived is not None:
            new_rowsel = convert_rs_to_rowsel_variable(rs, schema, n.row_sel_pred_derived)
            # for k,v in new_rowsel.items():
            #     print("{}\treplacing {} with {}".format(indent, k, str(v) if len(v) < 5 else '{}...'.format(v[:5])))
            rowsel_variable_values.update(new_rowsel)

    print("{}total lineage query time = {}".format(indent, total_time))
    for n in ppl_nodes:
        if isinstance(n.inference, SubqueryInference):
            run_lineage_query(orig_sql, n.inference.ppl_nodes, reference_data=rowsel_variable_values, subquery_level=subquery_level+1)
    
    
    return ret



