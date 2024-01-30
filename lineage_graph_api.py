from lineage_without_intermediate_result import lineage_inference_no_intermediate_result, predicate_pushup_pipeline
from get_spark_logical_plan import get_spark_logical_plan
from plan_ir_to_verify_core import convert_plan_to_pipeline, get_lineage_filter_from_query
from ir import UnresolvedRelation
from z3_eval_plan_ir import expr_to_sql, convert_rs_to_rowsel_variable, replace_rowsel_variables
import psycopg2

class LineageGraphNode:
    # base class
    def __init__(self):
        self.input_edges = set()
        self.output_edges = set()
        pass

    def add_input_edge(self, edge):
        self.input_edges.add(edge)

    def add_output_edge(self, edge):
        self.output_edges.add(edge)


class TableNode(LineageGraphNode):

    def __init__(self, table_name: str):
        super().__init__()
        self.name = table_name
        self.columns = [] # [('column_name','column_type')] #set()
        self.rows = []
        self.row_selection_sql = ""

    def add_column(self, column_name):
        self.columns.add(column_name)

    def add_row(self, row):
        self.rows.append(row)

    def set_row_selection_sql(self, row_selection: str):
        self.row_selection_sql = row_selection


class SQLNode(LineageGraphNode):
    def __init__(self, sql: str):
        super().__init__()

        # SQL Format: select ... from ...  into ...
        # (other operators, such as where/group by, are optional)
        self.sql = sql


class LineageGraphEdge:
    # directed edge
    def __init__(self, from_node: LineageGraphNode, to_node: LineageGraphNode):
        self.from_node = from_node
        self.to_node = to_node
        pass


class LineageGraph():
    # directed graph, no cycle
    # SQL Node and Table Node appear alternately
    def __init__(self):
        self.edges = []
        self.nodes = dict()
        pass

    def set_edges(self, edges):
        self.edges = edges

    def set_nodes(self, nodes):
        self.nodes = nodes

    def add_edge(self, edge: LineageGraphEdge):
        self.edges.append(edge)

    def add_node(self, node: LineageGraphNode, node_guid: str):
        self.nodes[node_guid] = node


# Engine API definition
"""
    Input:
    Lineage Graph, Entry Table GUID, Entry Row Selection SQL, Connection String(optional) 
    
    Output:
    After using PredTrace, fill each Table Node in the input lineage graph object with:
    - if connection string is provided: rows and row selection sql
        The format of a row:  sample_row={col_name1:value1,col_name2:value2......}  in which col_names and values are all strings
    - if connection string is not provided: row selection sql
    Finally the function will return the input lineage graph object.
"""
def row_level_lineage_query(graph: LineageGraph, entry_table_node_guid: str, entry_row_selection_sql: str,
                            connection: psycopg2.extensions.connection) -> LineageGraph:
#                            connection_string: str = "") -> LineageGraph:
    # TODO
    # return graph

    # set entry_row_selection_sql to the corresponding table
    # FIXME: how is guid used? now I assume it is tablename
    entry_table = None
    for n in graph.nodes:
        if isinstance(n, TableNode) and n.name == entry_table_node_guid:
            n.row_selection_sql = entry_row_selection_sql
            entry_table = n

    # sort SQL query nodes in topological order, stored in sql_nodes
    sql_nodes = []
    source_table_nodes = list(filter(lambda n: n not in [e.to_node for e in graph.edges], graph.nodes))
    processed_nodes = set(source_table_nodes)
    for edge in graph.edges:
        if isinstance(edge.to_node, TableNode):
            pass
        elif isinstance(edge.to_node, SQLNode):
            # all source tables for this SQL query is processed
            if edge.to_node not in processed_nodes and \
                all([e.from_node in processed_nodes for e in graph.edges if e.to_node==edge.to_node]):
                sql_nodes.append(edge.to_node)
        processed_nodes.add(edge.to_node)

    # pushdown to source tables
    for sql_node in reversed(sql_nodes):
        output_table = [e.to_node for e in graph.edges if e.from_node==sql_node][0]
        # skip sql where the output_table has no row_selection_sql
        # such sql whose input_tables has row_selection_sql is left later to push predicate up
        if output_table.row_selection_sql == '':
            continue
        input_tables = [e.from_node for e in graph.edges if e.to_node==sql_node]
        # get the SQL's logical query plan
        plan = get_spark_logical_plan(sql_node.sql)
        # convert logical plan into our internal pipeline representation 
        pipeline = convert_plan_to_pipeline(plan, {t.name:t.columns for t in input_tables})
        # get the filter represented in spark logical plan
        lineage_filter = get_lineage_filter_from_query(get_spark_logical_plan(output_table.row_selection_sql), output_table.columns) 
        # infer lineage
        pipeline_nodes = lineage_inference_no_intermediate_result(pipeline, lineage_filter)
        # retrieve lineage filter and set input_tables
        # because the input tables need to filter by specified order, sort them first
        pipeline_table_nodes = [n for n in pipeline_nodes if isinstance(n.plan_node, UnresolvedRelation)]
        pipeline_table_nodes.sort(key=lambda x:x.processing_order)
         
        rowsel_variable_values = {} # {variable:values}, used to store a mapping from row-sel-v? to actual row values; being set when only connection != None
        for n in pipeline_table_nodes:
            input_table = [t for t in input_tables if t.name==n.plan_node.tableIdentifier][0]
            if connection is not None:
                # replace row-sel-v? with concrete values
                sql_pred = expr_to_sql(replace_rowsel_variables(n.inference.input_filters[0], rowsel_variable_values))
            else:
                sql_pred = expr_to_sql(n.inference_input_filters[0])
            row_selection_sql = "SELECT * FROM {} WHERE {}".format(input_table.name, sql_pred)
            input_table.row_selection_sql = row_selection_sql
        
            if connection is not None:
                rows, schema = psql_run_query(connection, row_selection_sql, include_schema=True)
                input_table.rows = rows
                # print if it is source table without incoming graph edge (not intermediate able)
                if not any([e.to_node==input_table for e in graph.edges]):
                    print("table {}, row_selection_sql = {}, return {} rows\n".format(input_table.name, row_selection_sql, len(rows)))

                if hasattr(n, 'row_sel_pred_derived') and n.row_sel_pred_derived is not None:
                    # use the concrete rows to update row-sel-v?
                    new_rowsel = convert_rs_to_rowsel_variable(rows, schema, n.row_sel_pred_derived)
                    rowsel_variable_values.update(new_rowsel)

    # push predicate forward, find out rows affected at downstream
    for sql_node in sql_nodes:
        output_table = [e.to_node for e in graph.edges if e.from_node==sql_node][0]
        input_tables = [e.from_node for e in graph.edges if e.to_node==sql_node]
        if output_table.row_selection_sql != '':
            continue
        # can only push up if all input has row_selection_sql
        if not all([t.row_selection_sql!='' for t in input_tables]):
            continue
        # get the SQL's logical query plan
        plan = get_spark_logical_plan(sql_node.sql)
        # convert logical plan into our internal pipeline representation 
        pipeline = convert_plan_to_pipeline(plan, {t.name:t.columns for t in input_tables})
        # get the filter represented in spark logical plan
        lineage_filter = {t.name:\
                          get_lineage_filter_from_query(get_spark_logical_plan(t.row_selection_sql), t.columns)\
                          for t in input_tables}
        output_filter = predicate_pushup_pipeline(pipeline, lineage_filter)
        output_table.row_selection_sql = "SELECT * FROM {} WHERE {}".format(output_table.name, expr_to_sql(output_filter))
        print("Pushup downstream to table {}, row_selection_sql = {}\n".format(output_table.name, output_table.row_selection_sql))

    return graph


def psql_run_query(pgconn, sql, include_schema=False):
    cursor = pgconn.cursor()
    cursor.execute(sql)
    rs = cursor.fetchall()
    if include_schema:
        schema = [desc[0] for desc in cursor.description]
        return rs, schema
    else:
        return rs

                

        


