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
	s_name,
	count(*) as numwait
from
	supplier
inner join
	lineitem as l1 on s_suppkey = l1.l_suppkey
inner join
	orders on o_orderkey = l1.l_orderkey
inner join
	nation on s_nationkey = n_nationkey
where
	o_orderstatus = 'F'
	and l1.l_receiptdate > l1.l_commitdate
	and exists (
		select
			*
		from
			lineitem l2
		where
			l2.l_orderkey = l1.l_orderkey
			and l2.l_suppkey <> l1.l_suppkey
	)
	and not exists (
		select
			*
		from
			lineitem l3
		where
			l3.l_orderkey = l1.l_orderkey
			and l3.l_suppkey <> l1.l_suppkey
			and l3.l_receiptdate > l3.l_commitdate
	)
	and n_name = 'SAUDI ARABIA'
group by
	s_name
order by
	numwait desc,
	s_name
"""

# Supplier#000000496        |      17
lineage_filter_query = "SELECT * FROM output WHERE s_name='Supplier#000002540' AND numwait=17"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)

# FIXME: should not simplify predicate removing self-compared-row by only matching table name. Inner and outer table are the same table.

# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 