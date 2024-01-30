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
	sum(l_extendedprice* (1 - l_discount)) as revenue
from
	lineitem
inner join
	part on p_partkey = l_partkey
where
	(
		p_brand = 'Brand#12'
		and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
		and l_quantity >= 1 and l_quantity <= 1 + 10
		and p_size between 1 and 5
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)
	or
	(
		p_brand = 'Brand#23'
		and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
		and l_quantity >= 10 and l_quantity <= 10 + 10
		and p_size between 1 and 10
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)
	or
	(
		p_brand = 'Brand#34'
		and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
		and l_quantity >= 20 and l_quantity <= 20 + 10
		and p_size between 1 and 15
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	);
"""

# 3083843.0578
lineage_filter_query = "SELECT * FROM output WHERE revenue=3083843.0578"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)
# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 