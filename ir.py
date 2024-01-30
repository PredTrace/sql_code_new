import os
import sys
class PlanNode(object):
    def __init__(self, name, attrs={}):
        self.full_name = name
        self.name = name.split('.')[-1]
        self.children = []
        self.parent = None
        self.attrs = attrs
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    def to_s(self, level=0):
        indent = ''.join(['  ' for i in range(level)])
        s = indent + self.name #'(attr={}, {})\n'.format(','.join(['{}:{}'.format(k,str(v) if not isinstance(v, Dummy) else v.to_s()) for k,v in self.attr.items()]), len(self.children))
        attr_str = []
        for k,v in self.attrs.items():
            if isinstance(v, PlanNode):
                attr_str.append('{}:{}'.format(k, v.to_s(level+1)))
            elif type(v) is list:
                if len(v) > 0 and isinstance(v[0], PlanNode):
                    attr_str.append(str(k)+":"+''.join([v1.to_s(level+1) for v1 in v]))
                else:
                    attr_str.append('{}:{}'.format(k,v))
            else:
                attr_str.append('{}:{}'.format(k,v))
        s += ' attrs=({})'.format(','.join(attr_str))
        s += '\n'
        for c in self.children:
            s += c.to_s(level)
        return s
    def __str__(self):
        return self.name


# ========= org.apache.spark.sql.catalyst.plans.logical ==========
class LogicalPlan(object):
    def __str__(self):
        s = []
        for attr in self.__dir__():
            if attr.startswith('__') or attr in ['children'] or (attr=='child' and not isinstance(self, UnaryExpression)) or callable(getattr(self, attr)):
                continue
            s.append('{}={}'.format(attr, getattr(self, attr)))
        return '{}: ({})'.format(type(self).__name__, ','.join(s))
class UnaryNode(LogicalPlan):
    pass
class BinaryNode(LogicalPlan):
    pass
class SetOperation(LogicalPlan):
    pass
class LeafNode(LogicalPlan):
    pass

class Filter(UnaryNode):
    def __init__(self, condition, child):
        self.condition = condition # Expression
        self.child = child # LogicalPlan
        assert(isinstance(condition, Expression))
        assert(isinstance(child, LogicalPlan))

class Project(UnaryNode):
    def __init__(self, projectList, child):
        self.child = child # LogicalPlan
        self.projectList = projectList # Seq[Expression]
        assert(isinstance(child, LogicalPlan))
        assert(isinstance(projectList, list))

class UnresolvedHaving(UnaryNode):
    def __init__(self, condition, child):
        self.condition = condition # Expression
        self.child = child # LogicalPlan
        assert(isinstance(condition, Expression))
        assert(isinstance(child, LogicalPlan))

class Aggregate(UnaryNode):
    def __init__(self, groupingExpressions, aggregateExpressions, child):
        self.child = child # LogicalPlan
        self.groupingExpressions = groupingExpressions # Seq[Expression]
        self.aggregateExpressions = aggregateExpressions # Seq[NamedExpression]
        assert(isinstance(child, LogicalPlan))
        assert(isinstance(groupingExpressions, list))
        assert(isinstance(aggregateExpressions, list))

class Distinct(UnaryNode):
    def __init__(self, child):
        self.child = child # LogicalPlan
        assert(isinstance(child, LogicalPlan))

class SubqueryAlias(UnaryNode): # TODO
    def __init__(self, alias, child):
        self.alias = alias
        self.child = child # LogicalPlan
        assert(alias is None or isinstance(alias, str))
        assert(isinstance(child, LogicalPlan))

class GlobalLimit(UnaryNode):
    def __init__(self, limitExpr, child):
        self.limitExpr = limitExpr # Expression
        self.child = child # LogicalPlan
        assert(isinstance(limitExpr, Expression))
        assert(isinstance(child, LogicalPlan))

class LocalLimit(UnaryNode):
    def __init__(self, limitExpr, child):
        self.limitExpr = limitExpr # Expression
        self.child = child # LogicalPlan

class Intersect(SetOperation):
    def __init__(self, left, right):
        self.left = left # LogicalPlan
        self.right = right # LogicalPlan

class Join(BinaryNode):
    def __init__(self, left, right, joinType, condition):
        self.left = left # LogicalPlan
        self.right = right # LogicalPlan
        self.joinType = joinType
        self.condition = condition # Expression
        assert(isinstance(left, LogicalPlan))
        assert(isinstance(right, LogicalPlan))

class Range(LeafNode):
    def __init__(self, start, end, step, numSlices, output):
        self.start = start # Long
        self.end = end # Long
        self.step = step # Long
        self.numSlices = numSlices # Int
        self.output = output # [Attribute]

class Union(LogicalPlan):
    def __init__(self, children):
        self.children = children # [LogicalPlan]
        assert(all([isinstance(c, LogicalPlan) for c in self.children]))

class Sort(UnaryNode):
    def __init__(self, order, global_, child):
        self.order = order
        self.global_ = global_
        self.child = child # LogicalPlan

class Pivot(UnaryNode): # groupByExprs: Seq[NamedExpression], pivotColumn: Expression, pivotValues: Seq[Literal], aggregates: Seq[Expression], child: LogicalPlan
    def __init__(self, pivotColumn, pivotValues, aggregates, child):
        self.pivotColumn = pivotColumn
        self.pivotValues = pivotValues
        self.aggregates = aggregates
        self.child = child
class Unpivot(UnaryNode):
    def __init__(self, valueColumnNames, values, variableColumnName, child):
        self.valueColumnNames = valueColumnNames
        self.values = values
        self.variableColumnName = variableColumnName
        self.child = child

class InsertIntoStatement(UnaryNode):
    def __init__(self, table, userSpecifiedCols, child):
        self.table = table
        self.userSpecifiedCols = userSpecifiedCols 
        self.child = child

# ====== org.apache.spark.sql.catalyst.analysis =====
class UnresolvedAttribute(LogicalPlan):
    def __init__(self, nameParts):
        # convert to lower case
        self.nameParts = [n.lower() for n in nameParts] #nameParts # Seq[String]
        assert(all([isinstance(n, str) for n in nameParts]))
    def get_column_name(self):
        return self.nameParts[-1]
    def get_full_name(self):
        return '.'.join(self.nameParts)
class UnresolvedRelation(LogicalPlan):
    def __init__(self, tableIdentifier, alias=None):
        # convert to lower case
        self.tableIdentifier = tableIdentifier.lower()
        self.alias = alias # String
        assert(isinstance(tableIdentifier, str))
class UnresolvedFunction(LogicalPlan):
    def __init__(self, name, children, isDinstict):
        self.name = name # 
        self.children = children # Seq[Expression]
        self.isDistinct = isDinstict # boolean
        assert(isinstance(name, str))
        assert(all([isinstance(c, Expression) or isinstance(c, UnresolvedAttribute) for c in self.children]))
class UnresolvedStar(LogicalPlan):
    def __init__(self):
        pass


# ======= org.apache.spark.sql.catalyst.expressions =======
class Expression(LogicalPlan):
    pass
class BinaryExpression(Expression):
    pass
class UnaryExpression(Expression):
    pass
class SubqueryExpression(Expression):
    pass
class BinaryOperator(BinaryExpression):
    pass
class BinaryArithmetic(BinaryOperator):
    pass
class BinaryComparison(BinaryOperator):
    pass
class CaseWhenBase(Expression):
    pass
class LeafExpression(Expression):
    pass
class Alias(UnaryExpression):
    def __init__(self, child, name):
        self.child = child
        self.name = name # String
class Add(BinaryArithmetic):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class Subtract(BinaryArithmetic):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class Multiply(BinaryArithmetic):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class Divide(BinaryArithmetic):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class And(BinaryOperator):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class Or(BinaryOperator):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression

class CaseWhen(CaseWhenBase):
    def __init__(self, branches, elseValue):
        self.branches = branches # Seq[(Expression, Expression)]
        self.elseValue = elseValue # Option[Expression]
class Cast(UnaryExpression):
    def __init__(self, child, dataType):
        self.child = child # Expression
        self.dataType = dataType
class Exists(SubqueryExpression):
    def __init__(self, query, exprId):
        self.query = query # LogicalPlan
        self.exprId = exprId # ExprId
class ListQuery(SubqueryExpression): # subquery returning a list
    def __init__(self, query, exprId):
        self.query = query # LogicalPlan
        self.exprId = exprId # ExprId
class InSubquery(SubqueryExpression): # TODO: cannot find in org.apache.spark.
    def __init__(self, field, subquery):
        self.field = field
        self.query = subquery.query
class ScalarSubquery(SubqueryExpression): # subquery returning a scalar, same as ListQuery, appear as part of InSubquery or EqualTo/GreaterThan/...
    def __init__(self, query, exprId):
        self.query = query # LogicalPlan
        self.exprId = exprId # exprId
class GreaterThan(BinaryComparison):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class GreaterThanOrEqual(BinaryComparison):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class LessThan(BinaryComparison):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class LessThanOrEquan(BinaryComparison):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class EqualTo(BinaryComparison):
    def __init__(self, left, right):
        self.left = left # Expression
        self.right = right # Expression
class Like(BinaryComparison):
    def __init__(self, left, right):
        self.left = left
        self.right = right
class In(Expression):
    def __init__(self, value, list_):
        self.value = value # Expression
        self.list = list_ # Seq[Expression]
class Not(UnaryExpression):
    def __init__(self, child):
        self.child = child # Expression
class If(Expression):
    def __init__(self, predicate, trueValue, falseValue):
        self.predicate = predicate # Expression
        self.trueValue = trueValue # Expression
        self.falseValue = falseValue # Expression
class IsNULL(UnaryExpression):
    def __init__(self, child):
        self.child = child # Expression
class IsNotNULL(UnaryExpression):
    def __init__(self, child):
        self.child = child # Expression
class Coalesce(Expression):
    def __init__(self, children):
        self.children = children
class SortOrder(Expression):
    def __init__(self, direction, child):
        self.direction = direction
        self.child = child # expression
class Literal(LeafExpression):
    def __init__(self, value, dataType):
        self.value = value
        self.dataType = dataType
class Length(UnaryExpression):
    def __init__(self, child):
        self.child = child # Expression


def parse_list_string(s):
    return list([s1.lstrip() for s1 in s.replace('[','').replace(']','').split(',')])
def to_plan_ir(node):
    if not isinstance(node, PlanNode):
        return node
    ret = None
    if node.name == 'Project':
        ret = Project([to_plan_ir(c) for c in node.attrs['projectList']], to_plan_ir(node.children[0]))
    if node.name == 'Filter':
        ret = Filter(to_plan_ir(node.attrs['condition']), to_plan_ir(node.children[0]))
    if node.name == 'UnresolvedHaving':
        ret = UnresolvedHaving(to_plan_ir(node.attrs['havingCondition']), to_plan_ir(node.children[0]))
    if node.name == 'Aggregate':
        ret = Aggregate([to_plan_ir(c) for c in node.attrs['groupingExpressions']], [to_plan_ir(c) for c in node.attrs['aggregateExpressions']], to_plan_ir(node.children[0]))
    if node.name == 'Distinct':
        ret = Distinct(to_plan_ir(node.children[0]))
    if node.name == 'SubqueryAlias': 
        ret = SubqueryAlias(node.attrs['identifier']['name'] if not node.attrs['identifier']['name'].startswith('__auto') else None, to_plan_ir(node.children[0]))
    if node.name == 'GlobalLimit' or node.name == 'LocalLimit': 
        ret = GlobalLimit(to_plan_ir(node.attrs['limitExpr']), to_plan_ir(node.children[0]))
    if node.name == 'Sort': 
        ret = Sort([to_plan_ir(c) for c in node.attrs['order']], node.attrs['global'], to_plan_ir(node.children[0]))
    if node.name == 'Intersect':
        ret = Intersect(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'Union':
        ret = Union([to_plan_ir(c) for c in node.children])
    if node.name == 'Join':
        ret = Join(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]), node.attrs['joinType']['object'], \
                   to_plan_ir(node.attrs['condition']) if 'condition' in node.attrs else None)
    

    if node.name == 'UnresolvedAttribute':
        ret = UnresolvedAttribute(parse_list_string(node.attrs['nameParts']))
    if node.name == 'UnresolvedRelation':
        ret = UnresolvedRelation(parse_list_string(node.attrs['multipartIdentifier'])[0], None)
    if node.name == 'UnresolvedFunction':
        ret = UnresolvedFunction(parse_list_string(node.attrs['nameParts'])[0], [to_plan_ir(c) for c in node.children], node.attrs['isDistinct'])
    if node.name == 'UnresolvedStar':
        ret = UnresolvedStar()

    if node.name == 'Alias':
        ret = Alias(to_plan_ir(node.children[0]), node.attrs['name'])
    if node.name == 'Literal':
        v = node.attrs['value']
        if node.attrs['dataType'] == 'integer':
            v = int(v)
        ret = Literal(v, node.attrs['dataType'])
    if node.name == 'SortOrder':
        is_descending = 'Descending' in node.attrs['direction']['object'] 
        ret = SortOrder(is_descending, to_plan_ir(node.children[0]))
    if node.name == 'Add':
        ret = Add(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'Subtract':
        ret = Subtract(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'Multiply':
        ret = Multiply(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'Divide':
        ret = Divide(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'And':
        ret = And(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'Or':
        ret = Or(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'EqualTo':
        ret = EqualTo(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'Like':
        ret = Like(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'GreaterThan':
        ret = GreaterThan(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'GreaterThanOrEqual':
        ret = GreaterThanOrEqual(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'LessThan':
        ret = LessThan(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'LessThanOrEqual':
        ret = LessThanOrEquan(to_plan_ir(node.children[0]), to_plan_ir(node.children[1]))
    if node.name == 'In':
        ret = In(to_plan_ir(node.children[0]), [to_plan_ir(c) for c in node.children[1:]])
    if node.name == 'Not':
        ret = Not(to_plan_ir(node.children[0]))  
    if node.name == 'IsNull':
        ret = IsNULL(to_plan_ir(node.children[0]))
    if node.name == 'IsNotNull':
        ret = IsNotNULL(to_plan_ir(node.children[0]))
    if node.name == 'Coalesce':
        ret = Coalesce([to_plan_ir(c) for c in node.children])
    if node.name == 'CaseWhen': # TODO
        children = [to_plan_ir(c) for c in node.children]
        pairs = []
        for i in range(len(children)):
            if i%2 == 0 and i+1 < len(children):
                pairs.append((children[i],children[i+1]))
        elseValue = to_plan_ir(node.attrs['elseValue'])
        if elseValue is not None:
            ret = CaseWhen(pairs,elseValue)
    if node.name == 'Exists':
        subquery = to_plan_ir(node.attrs['plan'])
        ret = Exists(subquery, None)
    if node.name == 'ListQuery':
        subquery = to_plan_ir(node.attrs['plan'])
        ret = ListQuery(subquery, None)
    if node.name == 'InSubquery':
        left = to_plan_ir(node.children[0])
        right = to_plan_ir(node.children[1])
        ret = InSubquery(left, right)
    if node.name == 'ScalarSubquery':
        subquery = to_plan_ir(node.attrs['plan'])
        ret = ScalarSubquery(subquery, None)
    if node.name == 'UnresolvedAlias':
        ret = to_plan_ir(node.children[0])
    if node.name == 'Pivot':
        ret = Pivot(to_plan_ir(node.attrs['pivotColumn']), [to_plan_ir(v) for v in node.attrs['pivotValues']], [to_plan_ir(e) for e in node.attrs['aggregates']], to_plan_ir(node.children[0]))
    if node.name == 'Unpivot':
        ret = Unpivot(node.attrs['valueColumnNames'].replace('[','').replace(']','').split(','),\
                       [to_plan_ir(v) for v in node.attrs['values']], node.attrs['variableColumnName'],\
                        to_plan_ir(node.children[0]))
    if node.name == 'InsertIntoStatement':
        ret = to_plan_ir(node.children[0])
    if ret is None:
        print("{} unhandled".format(node.name))
        assert(False)
    
    return ret

        
    
