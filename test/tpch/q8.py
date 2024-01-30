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
	o_year,
	sum(case
		when nation = 'BRAZIL' then volume
		else 0
	end) / sum(volume) as mkt_share
from
	(
		select
			extract(year from o_orderdate) as o_year,
			l_extendedprice * (1 - l_discount) as volume,
			n2.n_name as nation
		from
			part
        inner join
			lineitem on p_partkey = l_partkey
        inner join
			supplier on s_suppkey = l_suppkey
		inner join
			orders on l_orderkey = o_orderkey
        inner join
			customer on o_custkey = c_custkey
        inner join
			nation as n1 on c_nationkey = n1.n_nationkey
        inner join 
			nation as n2 on s_nationkey = n2.n_nationkey
        inner join
			region on n1.n_regionkey = r_regionkey
		where
			r_name = 'AMERICA'
			and o_orderdate > '1995-01-01' 
            and o_orderdate < '1996-12-31'
			and p_type = 'ECONOMY ANODIZED STEEL'
	) as all_nations
group by
	o_year
order by
	o_year;
"""


#1995 | 0.03443589040665479743
lineage_filter_query = "SELECT * FROM output WHERE o_year=1995 AND mkt_share=0.03443589040665479743"
lineage_filter_query = "SELECT * FROM output WHERE o_year=1995"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)
# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 
