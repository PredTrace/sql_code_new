import z3

class UninterpretedFunctionMngr(object):
    def __init__(self):
        self.function_map = {}
    def add_function(self, fname, f):
        self.function_map[fname] = f
    def has_function(self, fname):
        return fname in self.function_map
    def get_function(self, fname):
        return self.function_map[fname]

# temp solution only for TPCH
class DateValueMapping(object):
    def __init__(self):
        self.date_to_int = {}
        self.int_to_date = {}
    def add_date(self, date, intv):
        self.date_to_int[date] = intv
        self.int_to_date[intv] = date
uninterpreted_funcion_mngr = UninterpretedFunctionMngr()
datevalue_mapping = DateValueMapping()

class AstRefKey:
    def __init__(self, n):
        self.n = n
    def __hash__(self):
        return self.n.hash()
    def __eq__(self, other):
        return self.n.eq(other.n)
    def __repr__(self):
        return str(self.n)

def askey(n):
    assert isinstance(n, z3.AstRef)
    return AstRefKey(n)

def get_vars_from_formula(f):
    r = set()
    def collect(f):
      if z3.is_const(f): 
          if f.decl().kind() == z3.Z3_OP_UNINTERPRETED and not askey(f) in r:
              r.add(askey(f))
      else:
          for c in f.children():
              collect(c)
    collect(f)
    return r

def z3_simplify(expr):
    if type(expr) in [bool, int, str, float]:
        return expr
    else:
        return z3.simplify(expr)

def check_always_hold(expr,debug_vars=[],eval_exprs={}):
    #print("CHECKING: {}".format(z3.simplify(expr)))
    variables = get_vars_from_formula(expr)
    additional_vars = []
    table_vars = []
    for v in variables:
        if str(v).startswith('additional_'):
            additional_vars.append(v.n)
        else:
            table_vars.append(v.n)

    solver = z3.Solver()
    solver.push()
    if len(additional_vars) == 0:
        solver.add(z3.Not(expr))
    else:
        solver.add(z3.Not(z3.Exists(additional_vars, z3.ForAll(table_vars, expr))))

    if solver.check() != z3.unsat:
        print("Solver failed!")
        for v in debug_vars:
            print("var {} = {}".format(v, solver.model()[v]))
        if len(eval_exprs) > 0:
            print("expr result:")
            for i,expr in eval_exprs.items():
                print("{} : {}".format(i,expr if type(expr) in [int, bool] else solver.model().eval(expr)))
        # solver.pop()
        return False
    else:
        solver.pop()
        return True
    
def list_union(*l):
    ret = l[0]
    for lst in l[1:]:
        ret = ret + lst
    return ret