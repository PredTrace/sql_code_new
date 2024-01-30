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
    nation,
    o_year,
    sum(amount) as sum_profit
from
    (
        select
            n_name as nation,
            extract(year from o_orderdate) as o_year,
            l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
        from
            orders
        inner join 
            lineitem on l_orderkey = o_orderkey
        inner join 
            part on p_partkey = l_partkey
        inner join
            partsupp on ps_partkey = p_partkey
        inner join
            supplier on s_suppkey = ps_suppkey
        inner join
            nation on s_nationkey = n_nationkey
        where
            ps_suppkey = l_suppkey
            and p_name like '%green%'
    ) as profit
group by
    nation,
    o_year
order by
    nation,
    o_year desc;
"""

sql = """
select
            n_name as nation,
            extract(year from o_orderdate) as o_year,
            l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
        from
            orders
        inner join 
            lineitem on l_orderkey = o_orderkey
        inner join 
            part on p_partkey = l_partkey
        inner join
            partsupp on ps_partkey = p_partkey
        inner join
            supplier on s_suppkey = ps_suppkey
        inner join
            nation on s_nationkey = n_nationkey
        where
            ps_suppkey = l_suppkey
            and p_name like '%green%'
"""

#BRAZIL                    |   1998 | 26527736.3960
lineage_filter_query = "SELECT * FROM output WHERE nation='ALGERIA' AND o_year=1998 AND sum_profit=27136900.1803"
lineage_filter_query = "SELECT * FROM output WHERE nation='BRAZIL' AND o_year=1998"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)

from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)
# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 
