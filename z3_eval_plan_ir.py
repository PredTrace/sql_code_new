import os
import sys
import z3
from ir import *
from util import uninterpreted_funcion_mngr, datevalue_mapping

def is_symbolic_expr(a):
    return isinstance(a, z3.ExprRef)  or isinstance(a, Value)
def is_constant_expr(a):
    return isinstance(a, int) or isinstance(a, str) or isinstance(a, bool) 
def getv(a):
    return a.v if isinstance(a, Value) else a
def compute_op(class_, *args):
    # TODO: handle NULL
    vs = [getv(a) for a in args]
    #print("{} / {}".format(class_, ','.join(['{}({})'.format(c, type(c)) for c in vs])))
    ret = operator_map[class_](*vs)
    return Value(ret)
    #return ret
builtin_aggr_functions = ['sum','mean','median','std','avg','count','variance','corr','any','every','min','max','distinct']

operator_map = {Add:lambda x,y:x+y, Subtract:lambda x,y:x-y, Multiply: lambda x,y:x*y, Divide: lambda x,y:x/y,\
               And:lambda x,y:z3.And(x, y), Or:lambda x,y:z3.Or(x, y), Not:lambda x:z3.Not(x),\
               GreaterThan: lambda x,y:x>y, GreaterThanOrEqual:lambda x,y:x>=y, EqualTo:lambda x,y:x==y, Like:lambda x,y:x==y, LessThan:lambda x,y:x<y, LessThanOrEquan:lambda x,y:x<=y,\
               If: lambda x,y,z:z3.If(x, y, z) if is_symbolic_expr(x) else y if x else z, \
               Length:lambda x:len(x), In: lambda x,*y: z3.Or(*[x==y_ for y_ in y])}

operator_to_sql_map = {Add:lambda x,y:'({} + {})'.format(x,y),\
                    Subtract:lambda x,y:'({} - {})'.format(x,y),\
                    Multiply:lambda x,y:'({} * {})'.format(x,y),\
                    Divide:lambda x,y:'({} / {})'.format(x,y),\
                    And:lambda x,y:'({} AND {})'.format(x,y),\
                    Or:lambda x,y:'({} OR {})'.format(x,y),\
                    Not:lambda x:'(NOT {})'.format(x),\
                    GreaterThan:lambda x,y:'{} > {}'.format(x,y),\
                    GreaterThanOrEqual:lambda x,y:'{} >= {}'.format(x,y),\
                    LessThan:lambda x,y:'{} < {}'.format(x,y),\
                    LessThanOrEquan:lambda x,y:'{} <= {}'.format(x,y),\
                    EqualTo:lambda x,y:'{} = {}'.format(x,y),\
                    Like:lambda x,y:'{} like {}'.format(x,y),\
                    Length:lambda x:'LENGTH({})'.format(x),\
                    In:lambda x,*y:'{} IN ({})'.format(x, ','.join([y_ for y_ in y]))}

import random
def eval_uninterpreted_function(expr, tup):
    fname = expr.name
    params = [getv(zeval(c, tup)) for c in expr.children]
    params = [z3.Const(e, z3.StringSort()) if type(e) is str else e for e in params]
    if uninterpreted_funcion_mngr.has_function(fname):
        newf = uninterpreted_funcion_mngr.get_function(fname)
    else:
        # FIXME set the correct return type...
        z3_param_types = [get_z3type_by_variable(p) for p in params]
        temp = list(filter(lambda x: not isinstance(x[0], Literal), [(p,i) for i,p in enumerate(expr.children)]))
        infer_return_type = [get_z3type_by_variable(params[i]) for p,i in temp]
        return_type = infer_return_type[0] #z3_param_types[0] 
        z3_param_types.append(return_type)
        newf = z3.Function('uninterpreted-{}-{}'.format(fname, random.randint(0, 100000)), *z3_param_types)
        uninterpreted_funcion_mngr.add_function(fname, newf)
    #print("params = {}".format(' , '.join(['{}:{}'.format(e, type(e)) for e in params])))
    retv = newf(*params)
    return Value(retv)


from dateutil import parser
# TODO: by default return a Value() object, which includes an indicator for NULL/NOT NULL
def zeval(expr, tup, additional=None):
    if is_symbolic_expr(expr):
        return expr
    elif is_constant_expr(expr):
        if type(expr) is str:
            return z3.Const(expr, z3.StringSort())
        else:
            return expr
    if isinstance(expr, Literal):
        if is_type_treated_as_bool(expr.dataType):
            return True if expr.value.lower()=='true' else False
        if is_type_treated_as_str(expr.dataType) and type(expr.value) is str:
            return z3.Const(expr.value, z3.StringSort())
        if is_type_treated_as_int(expr.dataType) and type(expr.value) is str:
            if '.' in expr.value:
                return int(float(expr.value))
            elif '-' in expr.value:
                return parser.parse(str(expr.value)).timestamp()
            else:
                return int(expr.value)
        return expr.value
    if isinstance(expr, UnresolvedAttribute):
        if tup.has_column(expr.get_full_name()):
            return zeval(tup[expr.get_full_name()], tup, additional)
        print("{} not in schema".format(expr.get_full_name()))
        print(",".join([k for k,v in tup.values.items()]))
        #print(expr, tup)
        assert(False)
    if isinstance(expr, UnresolvedFunction):
        children = [getv(zeval(expr.children[i], tup, {})) for i in range(len(expr.children))]
        if expr.name == 'any':
            val = children[0] if expr not in additional else z3.If(children[0], True, additional[expr])
        elif expr.name == 'every':
            val = children[0] if expr not in additional else z3.If(children[0], additional[expr], False)
        elif expr.name == 'max':
            val = children[0] if expr not in additional else z3.If(children[0]>additional[expr], children[0], additional[expr])
        elif expr.name == 'min':
            val = children[0] if expr not in additional else z3.If(children[0]<additional[expr], children[0], additional[expr])
        elif expr.name in ['sum','mean','median','std','avg','count','variance','corr']:
            #print("children = {}, expr = {}, additional = {}".format(children[0], expr, additional))
            val = children[0] if expr not in additional else children[0]+additional[expr]
        elif expr.name == 'distinct':
            val = children[0]
        else:
            #print("Process unhandled expression: {} / {}".format(expr.name, uninterpreted_funcion_mngr.has_function(expr.name)))
            val = eval_uninterpreted_function(expr, tup)
        if additional is not None:  
            additional[expr] = val
        return val
            
    if isinstance(expr, BinaryExpression):
        left = zeval(expr.left, tup, additional)
        right = zeval(expr.right, tup, additional)
        #return operator_map[expr.__class__](left, right)
        return compute_op(expr.__class__, left, right)
    if isinstance(expr, Not): # NULL/NOTNULL is different
        child = zeval(expr.child, tup, additional)
        #return operator_map[expr.__class__](child)
        return compute_op(expr.__class__, child)
    if isinstance(expr, In):
        return compute_op(expr.__class__, zeval(expr.value, tup, additional), *[zeval(e, tup, additional) for e in expr.list])
    if isinstance(expr, If):
        x = zeval(expr.predicate, tup, additional)
        y = zeval(expr.trueValue, tup, additional)
        z = zeval(expr.falseValue, tup, additional)
        #return operator_map[expr.__class__](x, y, z)
        return compute_op(expr.__class__, x, y, z)
    if isinstance(expr, CaseWhen):
        ret = getv(zeval(expr.elseValue, tup, additional))
        for k,v in expr.branches:
            ret = z3.If(getv(zeval(k, tup, additional)), getv(zeval(v, tup, additional)), ret)
        return ret
    if isinstance(expr, Alias):
        ret = zeval(expr.child, tup, additional)
        return ret
    if isinstance(expr, IsNotNULL):
        # FIXME: implement notnull
        ret = zeval(expr.child, tup, additional)
        if isinstance(ret, Value):
            return ret.isnull==False
        else:
            return True
    if isinstance(expr, Coalesce):
        # FIXME: implement coalesce
        return zeval(expr.children[0], tup, additional)
    if isinstance(expr, Exists):
        return True
    print(expr, type(expr))
    assert(False)
        

def expr_to_sql(expr):
    if is_constant_expr(expr):
        return str(expr) if type(expr) is not str else '"{}"'.format(expr)
    if isinstance(expr, Literal):
        #return str(expr.value) if expr.dataType != 'string' else '"{}"'.format(expr.value)
        if is_type_treated_as_str(expr.dataType) and type(expr.value) is str:
            return "'{}'".format(expr.value)
        else:
            # temp solution for datetime type in TPCH
            if expr.value in datevalue_mapping.int_to_date:
                return "date '{}'".format(datevalue_mapping.int_to_date[expr.value])
            else:
                return str(expr.value)
    if isinstance(expr, UnresolvedAttribute):
        return '.'.join(expr.nameParts)
        #return expr.get_column_name()
    if isinstance(expr, BinaryExpression):
        return operator_to_sql_map[expr.__class__](expr_to_sql(expr.left), expr_to_sql(expr.right))
    if isinstance(expr, Alias):
        #return expr_to_sql(expr.child) + ' AS {}'.format(expr.name)
        return expr_to_sql(expr.child)
    if isinstance(expr, IsNotNULL):
        return "{} IS NOT NULL".format(expr_to_sql(expr.child))
    if isinstance(expr, IsNULL):
        return "{} IS NULL".format(expr_to_sql(expr.child))
    if isinstance(expr, UnaryExpression):
        return operator_to_sql_map[expr.__class__](expr_to_sql(expr.child))
    if isinstance(expr, In):
        return operator_to_sql_map[expr.__class__](expr_to_sql(expr.value),*[expr_to_sql(c) for c in expr.list])
    if isinstance(expr, CaseWhen):
        s = 'CASE '
        for k,v in expr.branches:
            s += 'WHEN {} THEN {} '.format(expr_to_sql(k), expr_to_sql(v))
        if expr.elseValue is not None:
            s += "ELSE {} ".format(expr_to_sql(expr.elseValue))
        s += "END"
        return s
    if isinstance(expr, UnresolvedFunction):
        return '({}({}))'.format(expr.name, ','.join([expr_to_sql(c) for c in expr.children]))
    if isinstance(expr, If):
        return 'If {} THEN {} ELSE {}'.format(expr_to_sql(expr.predicate), expr_to_sql(expr.trueValue), expr_to_sql(expr.falseValue))
    if isinstance(expr, Coalesce):
        return "COALESCE ({})".format(','.join([expr_to_sql(e) for e in expr.children]))
    print("to_sql unhandled: {}".format(expr))
    assert(False)

def walk_expr_tree(node, is_stop_node=lambda x:False, handle_rs=lambda x:x, union_rs=lambda x,y:x): # handle_rs/union_rs are lambdas
    if is_stop_node(node) or is_constant_expr(node):
        return handle_rs(node)
    if isinstance(node, Literal):
        return handle_rs(node)
    elif isinstance(node, UnresolvedAttribute):
        return handle_rs(node)
    elif isinstance(node, UnresolvedStar):
        return handle_rs(node)
    elif isinstance(node, UnresolvedFunction):
        if len(node.children) == 0:
            return handle_rs(node)
        ret = walk_expr_tree(node.children[0], is_stop_node, handle_rs, union_rs)
        for c in node.children[1:]:
            ret = union_rs(ret, walk_expr_tree(c, is_stop_node, handle_rs, union_rs))
        return ret
    elif isinstance(node, BinaryExpression):
        left = walk_expr_tree(node.left, is_stop_node, handle_rs, union_rs)
        right = walk_expr_tree(node.right, is_stop_node, handle_rs, union_rs)
        return union_rs(left, right)
    elif isinstance(node, UnaryExpression):
        child = walk_expr_tree(node.child, is_stop_node, handle_rs, union_rs)
        return child
    elif isinstance(node, CaseWhen):
        ret = walk_expr_tree(node.elseValue, is_stop_node, handle_rs, union_rs)
        for k,v in node.branches:
            ret = union_rs(ret, walk_expr_tree(k, is_stop_node, handle_rs, union_rs))
            ret = union_rs(ret, walk_expr_tree(v, is_stop_node, handle_rs, union_rs))
        return ret
    elif isinstance(node, In):
        ret = walk_expr_tree(node.value, is_stop_node, handle_rs, union_rs)
        for e in node.list:
            ret = union_rs(ret, walk_expr_tree(e, is_stop_node, handle_rs, union_rs))   
        return ret 
    elif isinstance(node, If):
        x = walk_expr_tree(node.predicate, is_stop_node, handle_rs, union_rs)
        y = walk_expr_tree(node.trueValue, is_stop_node, handle_rs, union_rs)
        z = walk_expr_tree(node.falseValue, is_stop_node, handle_rs, union_rs)
        return union_rs(x, union_rs(y,z))
    elif isinstance(node, Exists) or isinstance(node, InSubquery) or isinstance(node, ScalarSubquery):
        return handle_rs(node)
    elif isinstance(node, Coalesce):
        ret = walk_expr_tree(node.children[0], is_stop_node, handle_rs, union_rs)
        for e in node.children[1:]:
            ret = union_rs(ret, walk_expr_tree(e, is_stop_node, handle_rs, union_rs))
        return ret
    print("walk tree {} / {}  not handled".format(node, type(node)))
    assert(False)

class Value(object): # v-1, v-2, v-3, v-4
    def __init__(self, value, isnull=False):
        self.v = value
        self.isnull = isnull
    def equals(self, v_):
        if isinstance(v_, Value):
            return z3.Or(z3.And(self.isnull==True, v_.isnull==True), z3.And(self.isnull==False, v_.isnull==False, self.v==v_.v))
        else:
            return self.v == v_
    def __eq__(self, v_):
        return self.equals(v_)
    def isnull(self):
        return self.isnull
    def __str__(self):
        if self.isnull==True:
            return 'null'
        elif self.isnull==False:
            return str(self.v)
        else:
            return "{}({})".format(self.v, self.isnull)
class Tuple(object):
    def __init__(self, values, exist_cond=True, count=1):
        self.values = values # {col_name: Value}
        self.exist_cond = exist_cond
        self.count = count
        self.abbr_column_values = {k.split('.')[-1]:v for k,v in values.items()}
    def eval_exist_cond(self):
        return getv(zeval(self.exist_cond, self))
    def has_column(self, col):
        return col in self.values or col.split('.')[-1] in self.abbr_column_values 
    def __getitem__(self, col):
        if col in self.values:
            return self.values[col]
        else:
            return self.abbr_column_values[col.split('.')[-1]]
    def __str__(self):
        c = self.eval_exist_cond()
        return 'values = [{}], exist_cond = {}'.format(','.join(['{}:{}'.format(k,v) for k,v in self.values.items()]), c if type(c) is bool else z3.simplify(c))

# SparkSQL types
# integer, date, interval month, string
def is_type_treated_as_int(typ):
    return typ in ['int','float','datetime','date','integer','double'] or typ.startswith('decimal') or typ.startswith('interval')
def is_type_treated_as_str(typ):
    return typ in ['str','string']
def is_type_treated_as_bool(typ):
    return typ in ['bool','boolean']
def type_match(typ1, typ2):
    if is_type_treated_as_int(typ1) and is_type_treated_as_int(typ2):
        return True
    if is_type_treated_as_str(typ1) and is_type_treated_as_str(typ2):
        return True
    return False

def get_init_value_by_type(typ):
    if is_type_treated_as_int(typ):
        return 0
    elif is_type_treated_as_str(typ):
        return ''
    else:
        return 0
        #assert(False)
   
def create_tuple(values, exist_cond=True, count=1):
    #if isinstance(values[next(iter(values))], Value):
    return Tuple({k:getv(v) for k,v in values.items()}, exist_cond, count)
    # if all([isinstance(v, Value) for k,v in values.items()]):
    #     return Tuple(values, exist_cond, count)
    # else:
    #     return Tuple({k:Value(v) if v is not None else Value(get_init_value_by_type(k), True) for k,v in values.items()}, exist_cond, count)

import random
def generate_row_selection_predicate(schema):
    r = random.randint(1, 100)
    return AllAnd(*[EqualTo(UnresolvedAttribute(col.split('.')), Literal(get_symbolic_value_by_type(ctyp, 'rowsel-v-{}-{}'.format(r, col.split('.')[-1])), ctyp)) for col,ctyp in schema])

def get_symbolic_value_by_type(typ, vname):
    if is_type_treated_as_int(typ):
        return z3.Int(vname)
    elif is_type_treated_as_str(typ):
        return z3.String(vname)
    elif is_type_treated_as_bool(typ):
        return z3.Bool(vname)
    else:
        assert(False, 'type not supported')

def get_z3type_by_type(typ):
    if is_type_treated_as_int(typ):
        return z3.IntSort()
    if is_type_treated_as_str(typ):
        return z3.StringSort()
    if is_type_treated_as_bool(typ):
        return z3.BoolSort()
    else:
        print("Type {} not supported".format(typ))
        assert(False)

def get_z3type_by_variable(v):
    v = getv(v)
    if isinstance(v, z3.BoolRef) or type(v) is bool:
        return z3.BoolSort()
    elif isinstance(v, z3.SeqRef) or type(v) is str:
        return z3.StringSort()
    elif isinstance(v, z3.ExprRef) or type(v) in [int,float]:
        return z3.IntSort()
    else:
        print("type {} : {} not supported".format(v, type(v)))
        assert(False)

def generate_symbolic_table(table_name, schema, Ntuples):
    ret = []
    for i in range(Ntuples):
        t = {k:get_symbolic_value_by_type(v, '{}-tup-{}-{}'.format(table_name, i+1, k)) for k,v in schema}
        t = create_tuple(t)
        ret.append(t)
    return ret

def get_columns_used(expr):
    return walk_expr_tree(expr, is_stop_node=lambda x:False, \
                   handle_rs=lambda x: [x.get_full_name()] if isinstance(x, UnresolvedAttribute) else [], \
                   union_rs=lambda x,y:x+y)

def expr_has_subquery(expr):
    return walk_expr_tree(expr, is_stop_node=lambda x: isinstance(x, SubqueryExpression), \
                        handle_rs=lambda x: True if isinstance(x, SubqueryExpression) else False,
                        union_rs=lambda x,y: x or y)
# FIXME: need better algo to retrieve subquery
def get_subquery(expr):
    return walk_expr_tree(expr, is_stop_node=lambda x: ((isinstance(x, BinaryComparison) and isinstance(x.right, SubqueryExpression))) \
                          or isinstance(x, Exists) or isinstance(x, InSubquery) \
                          or (isinstance(x, Not) and isinstance(x.child, SubqueryExpression)), \
                        handle_rs=lambda x: [x] if (isinstance(x, Exists) or isinstance(x, InSubquery) )\
                                        else ([x.right] if (isinstance(x, BinaryComparison) and isinstance(x.right, SubqueryExpression))\
                                        else ([x.child] if (isinstance(x, Not) and isinstance(x.child, SubqueryExpression)) else [])),
                        union_rs=lambda x,y: x+y)
def get_subquery_compare(expr):
    return walk_expr_tree(expr, is_stop_node=lambda x:((isinstance(x, BinaryComparison) and isinstance(x.right, SubqueryExpression))) \
                          or isinstance(x, Exists) or isinstance(x, InSubquery) \
                          or (isinstance(x, Not) and isinstance(x.child, SubqueryExpression)), \
                            handle_rs = lambda x: [x] if ((isinstance(x, BinaryComparison) and isinstance(x.right, SubqueryExpression))) \
                          or isinstance(x, Exists) or isinstance(x, InSubquery) \
                          or (isinstance(x, Not) and isinstance(x.child, SubqueryExpression)) else [],
                          union_rs=lambda x,y:x+y)
    
def expr_is_const(expr):
    return walk_expr_tree(expr, is_stop_node=lambda x:False, \
                          handle_rs=lambda x: True if (is_constant_expr(x)) else False,
                          union_rs=lambda x,y: x and y)

def projection_has_aggregation(expr):
    return walk_expr_tree(expr, is_stop_node=lambda x:isinstance(x, UnresolvedFunction) and x.name in builtin_aggr_functions,
                          handle_rs=lambda x:True if isinstance(x, UnresolvedFunction) and x.name in builtin_aggr_functions else False,
                          union_rs=lambda x,y: x or y)
def collect_aggregation(expr):
    return walk_expr_tree(expr, is_stop_node=lambda x:isinstance(x, UnresolvedFunction) and x.name in builtin_aggr_functions,
                          handle_rs=lambda x:[x] if isinstance(x, UnresolvedFunction) and x.name in builtin_aggr_functions else [],
                          union_rs=lambda x,y:x+y)

def get_new_column_name_from_expr(expr):
    c = walk_expr_tree(expr, is_stop_node=lambda x:isinstance(x, Alias), \
                          handle_rs=lambda x: x.name if isinstance(x, Alias) else None,\
                         union_rs=lambda x,y: y if x is None else x)
    if c is None:
        return get_columns_used(expr)[0]
    else:
        return c

def infer_type(expr, column_types={}): # FIXME, union_rs
    return walk_expr_tree(expr, is_stop_node=lambda x: isinstance(x,Literal) or isinstance(x,UnresolvedAttribute), \
                          handle_rs = lambda x: x.dataType if isinstance(x, Literal) else column_types[x.get_full_name()], \
                          union_rs = lambda x,y: 'str' if is_type_treated_as_str(x) and is_type_treated_as_str(y) else 'int')

def get_equivalent_pairs(expr):
    return walk_expr_tree(expr, is_stop_node=lambda x: isinstance(x, EqualTo), \
                          handle_rs = lambda x: [(x.left, x.right)] if isinstance(x, EqualTo) else [], \
                          union_rs = lambda x,y: x+y)


def is_group_expr_scalar_function(expr):
    is_scalar = walk_expr_tree(expr, is_stop_node=lambda x: isinstance(x, UnresolvedFunction), \
                               handle_rs=lambda x: False if (isinstance(x, UnresolvedFunction) and x.name in builtin_aggr_functions) else True,
                               union_rs=lambda x,y: x and y)
    return is_scalar

def compute_aggregate(expr, table):
    if is_group_expr_scalar_function(expr):
        return zeval(expr, table[0])
    aggr_rs_dict = {}
    rs = zeval(expr, table[0], additional=aggr_rs_dict)
    for tup in table[1:]:
        v = zeval(expr, tup, additional=aggr_rs_dict)
        rs = z3.If(tup.eval_exist_cond(), getv(v), getv(rs))
    return Value(rs) # Value(rs, z3.Any(*[tup.eval_exist_cond() for tup in table]))

def AllOr(*args):
    if len(args)==0:
        return True
    ret = args[0]
    for a in args[1:]:
        ret = Or(ret, a)
    return ret
def AllAnd(*args):
    if len(args)==0:
        return True
    ret = args[0]
    for a in args[1:]:
        ret = And(ret, a)
    return ret

# compare_to_replace can be a lambda function
def get_filter_replacing_compare(expr, compare_to_replace):
    if expr is None or is_constant_expr(expr):
        return expr
    if isinstance(expr, UnaryExpression) or isinstance(expr, BinaryExpression):
        if type(compare_to_replace) is dict:
            if expr in compare_to_replace:
                return True
            else:
                return expr
        elif compare_to_replace(expr):
            return True
        if isinstance(expr, BinaryExpression):
            return expr.__class__(get_filter_replacing_compare(expr.left, compare_to_replace), get_filter_replacing_compare(expr.right, compare_to_replace))
        else:
            return expr.__class__(get_filter_replacing_compare(expr.child, compare_to_replace))
    return expr

def get_filter_replacing_function(expr, function_to_replace):
    if expr is None or is_constant_expr(expr):
        return expr
    if isinstance(expr, UnaryExpression):
        return expr.__class__(get_filter_replacing_function(expr.child, function_to_replace))
    elif isinstance(expr, BinaryExpression):
        return expr.__class__(get_filter_replacing_function(expr.left, function_to_replace),get_filter_replacing_function(expr.right, function_to_replace))
    elif isinstance(expr, UnresolvedFunction):
        if expr in function_to_replace:
            return function_to_replace[expr]
        return expr
    elif isinstance(expr, UnresolvedAttribute) or isinstance(expr, LeafExpression):
        return expr
    elif isinstance(expr, ScalarSubquery) or isinstance(expr, InSubquery):
        return expr
    else:
        print("{} not handled".format(expr))
        assert(False)

def get_filter_replacing_field(expr, column_to_replace, return_conjunct=False):
    if expr is None or is_constant_expr(expr):
        return expr
    if isinstance(expr, UnresolvedAttribute):
        if return_conjunct:
            ret = []
            if expr.get_full_name() in column_to_replace:
                ret.append(column_to_replace[expr.get_full_name()])
            if expr.get_column_name() in column_to_replace:
                ret.append(column_to_replace[expr.get_column_name()])
            if expr.get_full_name()==expr.get_column_name() and expr.get_full_name() in set([k.split('.')[-1] for k,v in column_to_replace.items()]):
                for k,v in column_to_replace.items():
                    if k.split('.')[-1] == expr.get_column_name():
                        ret.append(v)
            if len(ret) > 0:
                return ret
            else:
                return expr
        else:
            if expr.get_full_name() in column_to_replace:
                return column_to_replace[expr.get_full_name()]
            elif expr.get_column_name() in column_to_replace:
                return column_to_replace[expr.get_column_name()]
            elif expr.get_full_name()==expr.get_column_name() and expr.get_full_name() in set([k.split('.')[-1] for k,v in column_to_replace.items()]):
                for k,v in column_to_replace.items():
                    if k.split('.')[-1] == expr.get_column_name():
                        return v
            else:
                return expr
    elif isinstance(expr, UnresolvedFunction):
        return UnresolvedFunction(expr.name, [get_filter_replacing_field(e, column_to_replace, return_conjunct) for e in expr.children], expr.isDistinct)
    elif isinstance(expr, Alias):
        return Alias(get_filter_replacing_field(expr.child, column_to_replace, return_conjunct),expr.name)
    elif isinstance(expr, UnaryExpression):
        return expr.__class__(get_filter_replacing_field(expr.child, column_to_replace, return_conjunct))
    elif isinstance(expr, BinaryExpression):
        l = get_filter_replacing_field(expr.left, column_to_replace, return_conjunct)
        r = get_filter_replacing_field(expr.right, column_to_replace, return_conjunct)
        if return_conjunct and isinstance(l, list):
            return AllAnd(*[expr.__class__(l1, r) for l1 in l])
        elif return_conjunct and isinstance(r, list):
            return AllAnd(*[expr.__class__(l, r1) for r1 in r])
        return expr.__class__(l,r)
    elif isinstance(expr, In):
        return In(get_filter_replacing_field(expr.value, column_to_replace), [get_filter_replacing_field(e, column_to_replace, return_conjunct) for e in expr.list])
    elif isinstance(expr, If):
        return If(get_filter_replacing_field(expr.predicate, column_to_replace),\
                  get_filter_replacing_field(expr.trueValue, column_to_replace),\
                  get_filter_replacing_field(expr.falseValue,column_to_replace))
    elif isinstance(expr, CaseWhen):
        return CaseWhen([(get_filter_replacing_field(b[0], column_to_replace), get_filter_replacing_field(b[1], column_to_replace, return_conjunct)) for b in expr.branches],\
                        get_filter_replacing_field(expr.elseValue, column_to_replace))
    elif isinstance(expr, LeafExpression):
        return expr
    else:
        print("{} not handled".format(expr))
        assert(False)
        return expr
    
def is_col_in_exist_columns(col, exist_columns):
    return (col in exist_columns or \
            ('.' not in col and col in [c.split('.')[-1] for c in exist_columns]) or \
            col.split('.')[-1] in exist_columns)
def get_filter_replacing_nonexist_column(expr, exist_columns):
    if expr is None:
        return expr
    if isinstance(expr, UnaryExpression):
        if isinstance(expr.child, UnresolvedAttribute):
            if any([not is_col_in_exist_columns(col, exist_columns) for col in get_columns_used(expr.child)]):
                return True
            else:
                return expr
        if isinstance(expr, Not) or isinstance(expr, IsNotNULL) or isinstance(expr, IsNULL):
            child = get_filter_replacing_nonexist_column(expr.child, exist_columns)
            if type(child) is bool and child==True and child != expr.child:
                return True
        else:
            if isinstance(expr, Alias):
                return Alias(get_filter_replacing_nonexist_column(expr.child, exist_columns), expr.name)
            else:
                return expr.__class__(get_filter_replacing_nonexist_column(expr.child, exist_columns))
    if isinstance(expr, BinaryComparison):
        # print(expr.left)
        # print(get_columns_used(expr.left))
        if any([(not is_col_in_exist_columns(col, exist_columns)) for col in get_columns_used(expr.left)]):
            return True
        if any([(not is_col_in_exist_columns(col, exist_columns)) for col in get_columns_used(expr.right)]):
            return True
        if isinstance(expr.left, SubqueryExpression) or isinstance(expr.right, SubqueryExpression):
            return True
        return expr.__class__(get_filter_replacing_nonexist_column(expr.left, exist_columns),\
                              get_filter_replacing_nonexist_column(expr.right, exist_columns))
    if isinstance(expr, In):
        if any([(not is_col_in_exist_columns(col, exist_columns)) for col in get_columns_used(expr.value)]):
            return True
        return expr
    if isinstance(expr, CaseWhen):
        if any([(not is_col_in_exist_columns(col, exist_columns)) for col in get_columns_used(expr.elseValue)]):
            return True
        for b,v in expr.branches:
            if any([(not is_col_in_exist_columns(col, exist_columns)) for col in get_columns_used(b)]):
                return True
            if any([(not is_col_in_exist_columns(col, exist_columns)) for col in get_columns_used(v)]):
                return True
        return expr
    if isinstance(expr, BinaryExpression):
        return expr.__class__(get_filter_replacing_nonexist_column(expr.left, exist_columns),\
                              get_filter_replacing_nonexist_column(expr.right, exist_columns))
    if isinstance(expr, Coalesce):
        if any([any([(not is_col_in_exist_columns(col, exist_columns)) for col in get_columns_used(e)])\
                for e in expr.children]):
            return True
        return expr
    if isinstance(expr, Exists) or isinstance(expr, InSubquery):
        return True
    # FIXME: other types
    return expr

def replace_rowsel_variables(p, rowsel_variables):
    if isinstance(p, BinaryComparison):
        if isinstance(p.right, Literal) and p.right.value in rowsel_variables:
            if len(rowsel_variables[p.right.value])==0:
                return True
            if isinstance(p, EqualTo):
                return In(p.left, [Literal(v, p.right.dataType) for v in rowsel_variables[p.right.value]])
            else:
                return AllAnd(*[p.__class__(p.left, Literal(v, p.right.dataType)) for v in rowsel_variables[p.right.value]])
                #assert(False)
        elif isinstance(p.left, Literal) and p.left.value in rowsel_variables:
            if isinstance(p, EqualTo):
                return In(p.right, [Literal(v, p.left.dataType) for v in rowsel_variables[p.left.value]])
            else:
                print(p)
                assert(False)
        else:
            return p
    elif isinstance(p, BinaryExpression):
        return p.__class__(replace_rowsel_variables(p.left, rowsel_variables), replace_rowsel_variables(p.right, rowsel_variables))
    elif isinstance(p, UnaryExpression):
        return p.__class__(replace_rowsel_variables(p.child, rowsel_variables))
    elif isinstance(p, In):
        #return In(p.value, [replace_rowsel_variables(l, rowsel_variables) for l in p.list])
        return p
    else:
        return p
        print(p)
        assert(False) 


def wrap_column_name(col):
    return UnresolvedAttribute(col.split('.'))

def get_conjunction_literals(p):
    if isinstance(p, And):
        return get_conjunction_literals(p.left) + get_conjunction_literals(p.right)
    else:
        return [p]
def simplify_conjunction(p):
    literals = get_conjunction_literals(p)
    if len(literals) == 1:
        return p
    new_literals = []
    temp = set()
    for l in literals:
        if expr_to_sql(l) not in temp:
            temp.add(expr_to_sql(l))
            new_literals.append(l)
    return AllAnd(*new_literals)
def simplify_pred(p):
    if isinstance(p, And) or isinstance(p, Or):
        l = simplify_pred(p.left)
        r = simplify_pred(p.right)
        if is_constant_expr(l):
            return r
        if is_constant_expr(r):
            return l
        if isinstance(p, And):
            return And(l, r)
        if isinstance(p, Or):
            return Or(l, r)
    return p

def convert_rs_to_rowsel_variable(rs, schema, rowsel_pred):
    equiv_pairs = get_equivalent_pairs(rowsel_pred)
    rowsel_variable_to_column = [(p2.value,p1.get_column_name()) for p1,p2 in equiv_pairs]
    rowsel_variable_values = {k:[] for k,v in rowsel_variable_to_column}
    for i in range(len(schema)):
        rowsel_variable = list(filter(lambda pair: pair[1].lower()==schema[i].lower(), rowsel_variable_to_column))
        if len(rowsel_variable) == 0:
            #print("Column {} not exist in rowsel".format(schema[i]))
            continue
        rowsel_variable = rowsel_variable[0][0]
        rowsel_variable_values[rowsel_variable] = [row[i] for row in rs]
    return rowsel_variable_values

# for TPCH, remove components from rowsel_variable of the table itself
# e.g., o_orderkey==rowsel-v-o_orderkey is removed
# but o_orderkey==rowsel-v-l_orderkey is kept
def simplify_pred_remove_selfrow(pred):
    def check_comparison(c):
        if isinstance(c, EqualTo) and \
            isinstance(c.left, UnresolvedAttribute) and isinstance(c.right, Literal) and \
            is_symbolic_expr(c.right.value) and c.left.get_column_name() in str(c.right.value):
            return True
        return False
    ret = get_filter_replacing_compare(pred, check_comparison)
    return ret

# TODO: propagate and obtain equivalent pairs as additional predicate
# some hacks for now, need better algo
def get_additional_equivalence_from_equiv_pairs(pairs, target_columns, table_columns):
    pairs_str = [(expr_to_sql(p1),expr_to_sql(p2)) for p1,p2 in pairs]
    # pairs_str2 = [(p1.get_full_name() if isinstance(p1, UnresolvedAttribute) else expr_to_sql(p1),\
    #                p2.get_full_name() if isinstance(p2, UnresolvedAttribute) else expr_to_sql(p2)) for p1,p2 in pairs]
    predicate_to_add = []
    for c in target_columns:
        c = wrap_column_name(c)
        equiv_set = set([c.get_full_name()])
        equiv_values = []
        set_sz = len(equiv_set)
        while True:
            for i,x in enumerate(pairs_str):
                # if (x[0] in equiv_set or pairs_str2[i][0] in equiv_set) and x[1] not in equiv_set and pairs_str2[i][1] not in equiv_set:
                if (is_col_in_exist_columns(x[0], equiv_set) or x[0] in equiv_set) and not (is_col_in_exist_columns(x[1], equiv_set) or x[1] in equiv_set):
                    equiv_set.add(x[1])
                    equiv_values.append(pairs[i][1])
                #elif (x[1] in equiv_set or pairs_str2[i][1] in equiv_set) and x[0] not in equiv_set and pairs_str2[i][0] not in equiv_set:
                elif (is_col_in_exist_columns(x[1], equiv_set) or x[1] in equiv_set) and not (is_col_in_exist_columns(x[0], equiv_set) or x[0] in equiv_set):
                    equiv_set.add(x[0])
                    equiv_values.append(pairs[i][0])
            if set_sz < len(equiv_set):
                set_sz = len(equiv_set)
            else:
                break
        #print("check for {}".format(c))
        for v in equiv_values:
            #print("*   * ** equal: {}".format(v))
            if any([(p1==c.get_column_name() and p2==expr_to_sql(v)) or (p2==c.get_column_name() and p1==expr_to_sql(v)) for p1,p2 in pairs_str]):
                continue
            if len(get_columns_used(v)) == 0 or is_col_in_exist_columns(v.get_full_name(), table_columns):
                predicate_to_add.append(EqualTo(c, v))
    return predicate_to_add


def get_additional_equivalence_from_equiv_pairs2(pairs, table_columns):
    predicate_to_add = []
    str_pairs = [(expr_to_sql(e1),expr_to_sql(e2)) for e1,e2 in pairs]
    for i,x in enumerate(pairs):
        p1,p2 = x
        lh = None
        all_in_left = any([is_col_in_exist_columns(col, table_columns) for col in get_columns_used(p1)])
        all_in_right = all([is_col_in_exist_columns(col, table_columns) for col in get_columns_used(p2)])
        if len(get_columns_used(p1))>0 and all_in_left:
            lh = p1
        if len(get_columns_used(p2))>0 and all_in_right:
            lh = p2
        if lh is None:
            continue
        equiv_set = set([p1, p2])
        equiv_str_set = set([str_pairs[i][0], str_pairs[i][1]])
        #print(" * * * *  $ equiv str set = {} {}".format(str_pairs[i][0], str_pairs[i][1]))
        last_round_sz = 1
        while len(equiv_set) > last_round_sz:
            last_round_sz = len(equiv_set)
            for j,y in enumerate(str_pairs):
                p1_, p2_ = y
                if p1_ not in equiv_str_set and p2_ in equiv_str_set:
                    equiv_set.add(pairs[j][0])
                    equiv_str_set.add(p1_)
                elif p1_ in equiv_str_set and p2_ not in equiv_str_set:
                    equiv_set.add(pairs[j][1])
                    equiv_str_set.add(p2_)
        for expr in equiv_set:
            if expr != lh and all([is_col_in_exist_columns(col, table_columns) for col in get_columns_used(expr)]):
                if not any([(expr_to_sql(lh)==p[0] and expr_to_sql(expr)==p[1]) or (expr_to_sql(lh)==p[1] and expr_to_sql(expr)==p[0]) for p in str_pairs]):
                    predicate_to_add.append(EqualTo(lh, expr))
    return predicate_to_add
