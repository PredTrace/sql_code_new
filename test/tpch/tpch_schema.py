from pyspark.sql.types import StructField, StructType, LongType, DoubleType, StringType, DateType

SCHEMA = dict(
    {
        "customer": StructType(
            [
                StructField("c_custkey", LongType(), False),
                StructField("c_name", StringType(), False),
                StructField("c_address", StringType(), False),
                StructField("c_nationkey", LongType(), False),
                StructField("c_phone", StringType(), False),
                StructField("c_acctbal", DoubleType(), False),
                StructField("c_mktsegment", StringType(), False),
                StructField("c_comment", StringType(), False),
            ]
        ),
        "lineitem": StructType(
            [
                StructField("l_orderkey", LongType(), False),
                StructField("l_partkey", LongType(), False),
                StructField("l_suppkey", LongType(), False),
                StructField("l_linenumber", LongType(), False),
                StructField("l_quantity", DoubleType(), False),
                StructField("l_extendedprice", DoubleType(), False),
                StructField("l_discount", DoubleType(), False),
                StructField("l_tax", DoubleType(), False),
                StructField("l_returnflag", StringType(), False),
                StructField("l_linestatus", StringType(), False),
                StructField("l_shipdate", DateType(), False),
                StructField("l_commitdate", DateType(), False),
                StructField("l_receiptdate", DateType(), False),
                StructField("l_shipinstruct", StringType(), False),
                StructField("l_shipmode", StringType(), False),
                StructField("l_comment", StringType(), False),
            ]
        ),
        "nation": StructType(
            [
                StructField("n_nationkey", LongType(), False),
                StructField("n_name", StringType(), False),
                StructField("n_regionkey", LongType(), False),
                StructField("n_comment", StringType(), False),
            ]
        ),
        "orders": StructType(
            [
                StructField("o_orderkey", LongType(), False),
                StructField("o_custkey", LongType(), False),
                StructField("o_orderstatus", StringType(), False),
                StructField("o_totalprice", DoubleType(), False),
                StructField("o_orderdate", DateType(), False),
                StructField("o_orderpriority", StringType(), False),
                StructField("o_clerk", StringType(), False),
                StructField("o_shippriority", LongType(), False),
                StructField("o_comment", StringType(), False),
            ]
        ),
        "part": StructType(
            [
                StructField("p_partkey", LongType(), False),
                StructField("p_name", StringType(), False),
                StructField("p_mfgr", StringType(), False),
                StructField("p_brand", StringType(), False),
                StructField("p_type", StringType(), False),
                StructField("p_size", LongType(), False),
                StructField("p_container", StringType(), False),
                StructField("p_retailprice", DoubleType(), False),
                StructField("p_comment", StringType(), False),
            ]
        ),
        "partsupp": StructType(
            [
                StructField("ps_partkey", LongType(), False),
                StructField("ps_suppkey", LongType(), False),
                StructField("ps_availqty", LongType(), False),
                StructField("ps_supplycost", DoubleType(), False),
                StructField("ps_comment", StringType(), False),
            ]
        ),
        "region": StructType(
            [
                StructField("r_regionkey", LongType(), False),
                StructField("r_name", StringType(), False),
                StructField("r_comment", StringType(), False),
            ]
        ),
        "supplier": StructType(
            [
                StructField("s_suppkey", LongType(), False),
                StructField("s_name", StringType(), False),
                StructField("s_address", StringType(), False),
                StructField("s_nationkey", LongType(), False),
                StructField("s_phone", StringType(), False),
                StructField("s_acctbal", DoubleType(), False),
                StructField("s_comment", StringType(), False),
            ]
        ),
    }
)

def transform_schema(s):
    return [(f.name, 'string' if f.dataType == StringType() else 'int') for f in s.fields]


TPCH_SCHEMA = {tbl_name: transform_schema(v) for tbl_name,v in SCHEMA.items()}

import psycopg2
tpch_pgconn = psycopg2.connect(database="testdb",user="test")
tpch_pgconn.autocommit = True