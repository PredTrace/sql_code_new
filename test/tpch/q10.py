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
	c_custkey,
	c_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue,
	c_acctbal,
	n_name,
	c_address,
	c_phone,
	c_comment
from
	customer
inner join 
	orders on c_custkey = o_custkey
inner join 
	lineitem on l_orderkey = o_orderkey
inner join 
	nation on c_nationkey = n_nationkey
where
	o_orderdate >= '1993-10-01'
	and o_orderdate < '1994-01-01'
	and l_returnflag = 'R'
group by
	c_custkey,
	c_name,
	c_acctbal,
	c_phone,
	n_name,
	c_address,
	c_comment
order by
	revenue desc
"""

# 57040 | Customer#000057040 | 734235.2455 |    632.87 | JAPAN                     | Eioyzjf4pp                               | 22-895-641-3466 | sits. slyly regular requests sleep alongside of the regular inst
lineage_filter_query = "SELECT * FROM output WHERE c_custkey=57040 AND c_name='Customer#000057040' AND revenue=734235.2455 AND c_acctbal=632.87 AND n_name='JAPAN' AND c_address='Eioyzjf4pp' AND c_phone='22-895-641-3466' AND c_comment='sits. slyly regular requests sleep alongside of the regular inst'"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)

# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 