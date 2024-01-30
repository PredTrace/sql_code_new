import os
import sys
sys.path.append('../../')
sys.path.append('../')
from verify_core_op import *
from plan_ir_to_verify_core import *
from get_spark_logical_plan import *
from predicate_pushdown import *
from test_util import get_row_lineage, get_lineage_from_queries
from tpch_schema import TPCH_SCHEMA, tpch_pgconn

sql = """
select
	n_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue
from
	customer
inner join 
	orders on c_custkey = o_custkey
inner join 
	lineitem on l_orderkey = o_orderkey
inner join 
	supplier on l_suppkey = s_suppkey
inner join
	nation on s_nationkey = n_nationkey 
inner join 
	region on n_regionkey = r_regionkey
where
	c_nationkey = s_nationkey
	and r_name = 'ASIA'
	and o_orderdate >= '1994-01-01'
	and o_orderdate < '1995-01-01' 
group by
	n_name
order by
	revenue desc;
"""

sql = """
select
	n_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue
from
	lineitem
inner join 
	orders on l_orderkey = o_orderkey
inner join
	customer on c_custkey = o_custkey
inner join
	supplier on l_suppkey = s_suppkey
inner join
	nation on s_nationkey = n_nationkey 
inner join 
	region on n_regionkey = r_regionkey
where
	c_nationkey = s_nationkey
	and r_name = 'ASIA'
	and o_orderdate >= '1994-01-01'
	and o_orderdate < '1995-01-01' 
group by
	n_name
order by
	revenue desc;
"""


#INDONESIA                 | 55502041.1697
lineage_filter_query = "SELECT * FROM output WHERE n_name='INDONESIA' AND revenue=55502041.1697"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)
# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 