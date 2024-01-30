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
	100.00 * sum(case
		when p_type like 'PROMO%'
			then l_extendedprice * (1 - l_discount)
		else 0
	end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
	lineitem
inner join
	part on l_partkey = p_partkey
where
	l_shipdate >= '1995-09-01'
	and l_shipdate < '1995-10-01'
"""

#  16.3807786263955401
lineage_filter_query = "SELECT * FROM output WHERE promo_revenue=16.3807786263955401"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)

# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 


# lineage_filter = get_lineage_filter_from_query(get_spark_logical_plan(lineage_filter_query))
# ppl_nodes = predicate_pushdown_pipeline(ppl, lineage_filter)