import os
import sys
import shutil
from pyspark.sql import SparkSession
import json
import pprint
from ir import PlanNode, to_plan_ir

class SQLPlanAnalyzer:
    
    def __init__(self):
        pass

    def handle_children(self, name, node):
        if not name in node:
            assert(False)
        if type(node[name]) is list:
            return [self.deserialize_plan_node(expr)[0] for expr in node[name]]
        elif type(node[name]) is dict:
            # TODO
            return node[name]
        else:
            return node[name]
    def deserialize_plan_node(self, plan, it=0):
        if isinstance(plan, list):
            node = plan[it]
        elif isinstance(plan, dict):
            node = plan
        else:
            return plan, it + 1

        #print("{} / {}".format(type(node), str(node)[:100]))
        class_name = node.get("class")
        if class_name is None:
            return None, it
        d = PlanNode(class_name.split('.')[-1])
        for i in range(node.get('num-children')):
            child, it = self.deserialize_plan_node(plan, it + 1)
            d.add_child(child)
        attrs = {}
        #print("{} / {} / {} ".format(class_name, type(plan), it))
        for k,v in node.items():
            if k not in ['class','num-children','options'] and type(v) != list:
                attrs[k] = v
            elif type(v) is list:
                if len(v) > 0 and type(v[0]) is list:
                    # FIXME: the if case is only to handle nested list in Unpivot
                    if type(v[0][0]) is list and len(v[0])==1:
                        attrs[k] = [self.deserialize_plan_node(expr[0])[0] for expr in v]
                    else:
                        attrs[k] = [self.deserialize_plan_node(expr)[0] for expr in v]
                elif len(v) > 0 and type(v[0]) is dict:
                    attrs[k] = self.deserialize_plan_node(v)[0]
                else:
                    attrs[k] = v
        d.attrs = attrs
        return d, it
    def deserialize_plan(self, plan):
        it = 0
        node, it = self.deserialize_plan_node(plan)
        return node

def get_spark_logical_plan(query):
    spark = (
        SparkSession.builder.appName("LineageApp")
        .getOrCreate()
    )     
    # for tbl_name, schema in tpch_schema.SCHEMA.items():
    #     csv_df = (
    #         spark.read.options(header="true", delimiter="|")
    #         .schema(schema)
    #         .csv(f"./tpch-data/{tbl_name}.csv")
    #     )
    
    # # remove spark warehouse data
    # if os.path.exists("./spark-warehouse"):
    #     shutil.rmtree("./spark-warehouse")

    spark.sparkContext.setLogLevel(logLevel="ERROR")

    # save as managed table
    # csv_df.printSchema()
    # csv_df.write.saveAsTable(tbl_name)
    parser = spark._jsparkSession.sessionState().sqlParser()
    plan = parser.parsePlan(query)

    plan_json = json.loads(plan.toJSON())
    #pprint.pprint(plan_json)

    temp = SQLPlanAnalyzer().deserialize_plan_node(plan_json)[0]
    ret = (to_plan_ir(temp))

    return ret



query = """select revenue, o.c_custkey,o.c_acctbal,o.c_address, o.c_phone,o.c_comment, o.n_name,o.c_name,o_orderkey,l_linenumber,n_nationkey
from
(select
        c_custkey,
        c_name,
        sum(l_extendedprice * (1 - l_discount)) as revenue,
        c_acctbal,
        n_name,
        c_address,
        c_phone,
        c_comment
from
        customer,
        orders,
        lineitem,
        nation
where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate >= date '1993-10-01'
        and o_orderdate < date '1993-10-01' + interval '3' month
        and l_returnflag = 'R'
        and c_nationkey = n_nationkey
group by
        c_custkey,
        c_name,
        c_acctbal,
        c_phone,
        n_name,
        c_address,
        c_comment
order by
        revenue desc
LIMIT 20);
"""
    





        
        
        
    
    
#print(ret[0].to_s())


