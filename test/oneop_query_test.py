import os
import sys
sys.path.append('../')
from verify_core_op import *
from plan_ir_to_verify_core import *
from get_spark_logical_plan import *
from predicate_pushdown import *
from test_util import get_row_lineage

def test_filter1():
    table_schemas = {'tableA':[('col1','int'),('col2','int')]}
    sql = "SELECT * FROM tableA AS A WHERE A.col1=1"

    lineage_filter_query = "SELECT * FROM output WHERE col2=2"
    get_row_lineage(table_schemas, sql, lineage_filter_query)

def test_filter2():
    table_schemas = {'tableA':[('col1','int'),('col2','int')]}
    sql = "SELECT col1 FROM tableA WHERE col1!=1"
    
    lineage_filter_query = "SELECT * FROM output WHERE col1=2"
    get_row_lineage(table_schemas, sql, lineage_filter_query)

def test_projection():
    table_schemas = {'tableA':[('col1','int'),('col2','int')]}
    sql = "SELECT col1/col2 as c FROM tableA WHERE col1!=1"

    lineage_filter_query = "SELECT * FROM output WHERE c=2"
    get_row_lineage(table_schemas, sql, lineage_filter_query)

def test_innerjoin1():
    table_schemas = {'tableA':[('col1','int'),('col2','int')], 'tableB':[('col1','int'),('col4','int')]}
    sql = 'SELECT tableA.col1 AS c_1, tableB.col4 as c_2 FROM tableA INNER JOIN tableB ON tableA.col1 = tableB.col1'

    lineage_filter_query = "SELECT * FROM output WHERE c_2=10 AND c_1=20"
    get_row_lineage(table_schemas, sql, lineage_filter_query)

def test_innerjoin2():
    table_schemas = {'tableA':[('col1','int'),('col2','int')], 'tableB':[('col3','int'),('col4','int')]}
    sql = 'SELECT tableA.col1 AS c_1, tableB.col4 as c_2 FROM tableA INNER JOIN tableB ON tableA.col1 = tableB.col3'

    lineage_filter_query = "SELECT * FROM output WHERE c_2=10 AND c_1=20"
    get_row_lineage(table_schemas, sql, lineage_filter_query)

def test_groupby1():
    table_schemas = {'tableA':[('col1','int'),('col2','int')]}
    sql = "SELECT col1, sum(col2) AS sum_col FROM tableA GROUP BY col1"

    lineage_filter_query = "SELECT * FROM output WHERE col1=1 and sum_col=10"
    get_row_lineage(table_schemas, sql, lineage_filter_query)

test_groupby1()