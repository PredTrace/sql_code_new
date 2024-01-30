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
	l_shipmode,
	sum(case
		when o_orderpriority = '1-URGENT'
			or o_orderpriority = '2-HIGH'
			then 1
		else 0
	end) as high_line_count,
	sum(case
		when o_orderpriority <> '1-URGENT'
			and o_orderpriority <> '2-HIGH'
			then 1
		else 0
	end) as low_line_count
from
	orders
inner join 
	lineitem on o_orderkey = l_orderkey
where
	l_shipmode in ('MAIL', 'SHIP')
	and l_commitdate < l_receiptdate
	and l_shipdate < l_commitdate
	and l_receiptdate >= '1994-01-01'
	and l_receiptdate < '1995-01-01' 
group by
	l_shipmode
order by
	l_shipmode;
"""

#  MAIL       |            6202 |           9324
lineage_filter_query = "SELECT * FROM output WHERE l_shipmode='MAIL' AND high_line_count=6202 AND low_line_count=9324"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)

# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 