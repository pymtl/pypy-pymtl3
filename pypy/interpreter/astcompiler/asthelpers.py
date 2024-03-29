from pypy.interpreter.astcompiler import ast, misc
from pypy.interpreter.error import OperationError


class UnacceptableExpressionContext(Exception):

    def __init__(self, node, msg):
        self.node = node
        self.msg = msg
setattr(ast, "UnacceptableExpressionContext", UnacceptableExpressionContext)


class __extend__(ast.AST):

    def as_node_list(self, space):
        raise AssertionError("only for expressions")

    def set_context(self, space, ctx):
        raise AssertionError("should only be on expressions")


class __extend__(ast.expr):

    constant = False
    _description = None

    def as_node_list(self, space):
        return None

    def set_context(self, space, ctx):
        d = self._description
        if d is None:
            d = "%r" % (self,)
        if ctx == ast.Del:
            msg = "can't delete %s" % (d,)
        else:
            msg = "can't assign to %s" % (d,)
        raise UnacceptableExpressionContext(self, msg)


class __extend__(ast.List):

    def as_node_list(self, space):
        return self.elts

    def set_context(self, space, ctx):
        if self.elts:
            for elt in self.elts:
                elt.set_context(space, ctx)
        self.ctx = ctx


class __extend__(ast.Attribute):

    def set_context(self, space, ctx):
        if ctx == ast.Store:
            misc.check_forbidden_name(space, self.attr, self)
        self.ctx = ctx


class __extend__(ast.Subscript):

    def set_context(self, space, ctx):
        self.ctx = ctx


class __extend__(ast.Name):

    def set_context(self, space, ctx):
        if ctx == ast.Store:
            misc.check_forbidden_name(space, self.id, self)
        self.ctx = ctx


class __extend__(ast.Tuple):

    _description = "()"

    def as_node_list(self, space):
        return self.elts

    def set_context(self, space, ctx):
        if self.elts:
            for elt in self.elts:
                elt.set_context(space, ctx)
        self.ctx = ctx

class __extend__(ast.Lambda):

    _description = "lambda"


class __extend__(ast.Call):

    _description = "function call"


class __extend__(ast.BoolOp, ast.BinOp, ast.UnaryOp):

    _description = "operator"


class __extend__(ast.GeneratorExp):

    _description = "generator expression"


class __extend__(ast.Yield):

    _description = "yield expression"


class __extend__(ast.ListComp):

    _description = "list comprehension"


class __extend__(ast.SetComp):

    _description = "set comprehension"


class __extend__(ast.DictComp):

    _description = "dict comprehension"


class __extend__(ast.Dict, ast.Set, ast.Str, ast.Bytes, ast.Num, ast.Constant):

    _description = "literal"


class __extend__(ast.Compare):

    _description = "comparison"

class __extend__(ast.Starred):

    _description = "starred expression"

    def set_context(self, space, ctx):
        self.ctx = ctx
        self.value.set_context(space, ctx)

class __extend__(ast.IfExp):

    _description = "conditional expression"


class __extend__(ast.Constant):

    constant = True

    def as_node_list(self, space):
        try:
            values_w = space.unpackiterable(self.value)
        except OperationError:
            return None
        line = self.lineno
        column = self.col_offset
        return [ast.Constant(w_obj, line, column) for w_obj in values_w]


class __extend__(ast.Str):

    constant = True


class __extend__(ast.Num):

    constant = True


class __extend__(ast.Ellipsis):

    _description = "Ellipsis"
    constant = True


class __extend__(ast.NameConstant):

    _description = "name constant"
    constant = True
