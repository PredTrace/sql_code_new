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
	l_orderkey,
	sum(l_extendedprice * (1 - l_discount)) as revenue,
	o_orderdate,
	o_shippriority
from
	customer
inner join 
	orders on c_custkey = o_custkey
inner join 
	lineitem on l_orderkey = o_orderkey
where
	c_mktsegment = 'BUILDING'
	and o_orderdate < '1995-03-15'
	and l_shipdate > '1995-03-15'
group by
	l_orderkey,
	o_orderdate,
	o_shippriority
order by
	revenue desc,
	o_orderdate
"""

# 2456423 | 406181.0111 | 1995-03-05  |              0
lineage_filter_query = "SELECT * FROM output WHERE l_orderkey=2456423 AND revenue=406181.0111 AND o_orderdate='1995-03-05' AND o_shippriority=0"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)

# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 