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
	s_acctbal,
	s_name,
	n_name,
	p_partkey,
	p_mfgr,
	s_address,
	s_phone,
	s_comment
from
	part
inner join 
	partsupp on	p_partkey = ps_partkey
inner join
	supplier on s_suppkey = ps_suppkey
inner join
	nation on s_nationkey = n_nationkey
inner join
	region on n_regionkey = r_regionkey
where
	p_size = 15
	and p_type like '%BRASS'
	and r_name = 'EUROPE'
	and ps_supplycost = (
		select
			min(ps_supplycost)
		from
			partsupp
        inner join
			supplier on s_suppkey = ps_suppkey
        inner join
			nation on s_nationkey = n_nationkey
        inner join
			region on n_regionkey = r_regionkey
		where
			p_partkey = ps_partkey
			and r_name = 'EUROPE'
	)
order by
	s_acctbal desc,
	n_name,
	s_name,
	p_partkey
"""


#9938.53 | Supplier#000005359        | UNITED KINGDOM            |    185358 | Manufacturer#4            | QKuHYh,vZGiwu2FWEJoLDx04                 | 33-429-790-6131 | uriously regular requests hag
lineage_filter_query = "SELECT * FROM output WHERE s_acctbal=9938.53 AND s_name='Supplier#000005359' AND n_name='UNITED KINGDOM' AND p_partkey=185358 AND p_mfgr='Manufacturer#4' AND s_address='QKuHYh,vZGiwu2FWEJoLDx04' AND s_phone='33-429-790-6131' AND s_comment='uriously regular requests hag' "
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)
# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 