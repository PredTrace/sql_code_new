
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import psycopg2

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)

# connect to TPCH database
pgconn = psycopg2.connect(database="testdb",user="test")
pgconn.autocommit = True


