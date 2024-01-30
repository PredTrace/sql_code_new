 # Efficient Row-Level Lineage Leveraging Predicate Pushdown

 Row-level lineage explains what input rows produce an output row through a workflow, having many applications like data debugging, auditing, data integration, etc. 
In this repo, we introduce PredTrace, a lineage tracking system that achieves easy adaptation, low runtime overhead, efficient lineage querying, and high query/pipeline coverage. It achieves this by leveraging predicate pushdown: pushing a row-selection predicate that describes the target output down to source tables, and selects the lineage by running the pushed predicate.

We offer two options in PredTrace: one option slightly alters the original query/pipeline by saving intermediate results and returns the precise lineage; and another version without using intermediate results but may return a superset.


## Prerequisites
To install the packages and prepare for use, run:
```bash
$ git clone https://github.com/PredTrace/predtrace_sql.git
$ pip install -r requirements.txt
```

The following python packages are required to run PredTrace and z3.
1. z3 (https://github.com/Z3Prover/z3)
2. pyspark==3.4.1

##  Run TPCH:
```bash
$ cd test/tpch/
$ python q4.py
```

This will print out the lineage query predicate to process on each table.

You may notice some delay before seeing the lineage result which may differ from the inference time reported. This is because the delay is due to Spark SQL parser. When counting the inference time, we skip the parsing and start counting from the logical plan. 


### Connecting to databases
Setup the connection to database by updating test/db\_connection.py.
Then uncomment the last two lines 

```python
from eval_query import run_lineage_query
run_lineage_query(sql, ppl_nodes)
```
Run the lineage queries on source tables and view the lineage results.

### Obtaining exact lineage/lineage superset
Turn the prameter in `get_row_lineage` function (`no_intermediate_result=True` or `False`) to see the result of not saving / saving intermediate result.


### Specifying your own lineage queries
You may change the row to query by updating the row-selection filter in `lineage_filter_query`.



You may notice some delay before seeing the lineage result which may differ from the inference time reported. This is because the delay is due to Spark SQL parser. When counting the inference time, we skip the parsing and start counting from the logical plan.