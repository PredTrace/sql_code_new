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
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice,
    sum(l_quantity) as sum_quantity
from
    customer
inner join
    orders on c_custkey = o_custkey
inner join
    lineitem on o_orderkey = l_orderkey
where
    o_orderkey in (
        select
            l_orderkey
        from
            lineitem
        group by
            l_orderkey having
                sum(l_quantity) > 300
    )
group by
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice
order by
    o_totalprice desc,
    o_orderdate
"""


#  Customer#000128120 |    128120 |    4722021 | 1994-04-07  |    544089.09 | 323.00
lineage_filter_query = "SELECT * FROM output WHERE c_name='Customer#000128120' AND c_custkey=128120 AND o_orderkey=4722021 AND o_orderdate='1994-04-07' AND o_totalprice=544089.09 AND sum_quantity=323.00"
ppl_nodes = get_row_lineage(TPCH_SCHEMA, sql, lineage_filter_query, no_intermediate_result=True)
from lineage_without_intermediate_result2 import retrieve_lineage_data
retrieve_lineage_data(ppl_nodes[-1], tpch_pgconn)

# from eval_query import run_lineage_query
# run_lineage_query(sql, ppl_nodes) 