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
	ps_partkey,
	sum(ps_supplycost * ps_availqty) as value
from
	partsupp
inner join
	supplier on ps_suppkey = s_suppkey
inner join
	nation on s_nationkey = n_nationkey
where
	n_name = 'GERMANY'
group by
	ps_partkey having
		sum(ps_supplycost * ps_availqty) > (
			select
				sum(ps_supplycost * ps_availqty) * 0.0001000000
			from
				partsupp
            inner join
				supplier on ps_suppkey = s_suppkey
            inner join
				nation on s_nationkey = n_nationkey
			where
				n_name = 'GERMANY'
		)
order by
	value desc;
"""

#      129760 | 17538456.86
lineage_filter_query = "SELECT * FROM output WHERE ps_partkey=129760 AND value=17538456.86"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)

# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 