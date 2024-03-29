from rpython.rtyper.lltypesystem import rffi, lltype
from pypy.interpreter.error import OperationError, oefmt
from pypy.objspace.std.listobject import W_ListObject
from pypy.module._hpy_universal.apiset import API
from pypy.module._hpy_universal import handles

@API.func("HPy HPyList_New(HPyContext ctx, HPy_ssize_t len)")
def HPyList_New(space, ctx, len):
    if len == 0:
        w_list = space.newlist([])
    else:
        w_list = space.newlist([None] * len)
    return handles.new(space, w_list)

@API.func("int HPyList_Check(HPyContext ctx, HPy h)")
def HPyList_Check(space, ctx, h):
    w_obj = handles.deref(space, h)
    w_obj_type = space.type(w_obj)
    res = (space.is_w(w_obj_type, space.w_list) or
           space.issubtype_w(w_obj_type, space.w_list))
    return API.int(res)

@API.func("int HPyList_Append(HPyContext ctx, HPy h_list, HPy h_item)")
def HPyList_Append(space, ctx, h_list, h_item):
    w_list = handles.deref(space, h_list)
    # XXX the tests should check what happens in this case
    assert isinstance(w_list, W_ListObject)
    w_item = handles.deref(space, h_item)
    w_list.append(w_item)
    return API.int(0)
