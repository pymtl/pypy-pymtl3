from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.rlib import rutf8
from pypy.interpreter.error import OperationError, oefmt
from pypy.interpreter.unicodehelper import wcharpsize2utf8
from pypy.objspace.std import unicodeobject
from pypy.module._hpy_universal.apiset import API
from pypy.module._hpy_universal import handles

def _maybe_utf8_to_w(space, utf8):
    # should this be a method of space?
    s = rffi.constcharp2str(utf8)
    try:
        length = rutf8.check_utf8(s, allow_surrogates=False)
    except rutf8.CheckError:
        raise   # XXX do something
    return space.newtext(s, length)

@API.func("int HPyUnicode_Check(HPyContext ctx, HPy h)")
def HPyUnicode_Check(space, ctx, h):
    w_obj = handles.deref(space, h)
    w_obj_type = space.type(w_obj)
    res = (space.is_w(w_obj_type, space.w_unicode) or
           space.issubtype_w(w_obj_type, space.w_unicode))
    return API.int(res)

@API.func("HPy HPyUnicode_FromString(HPyContext ctx, const char *utf8)")
def HPyUnicode_FromString(space, ctx, utf8):
    w_obj = _maybe_utf8_to_w(space, utf8)
    return handles.new(space, w_obj)

@API.func("HPy HPyUnicode_AsUTF8String(HPyContext ctx, HPy h)")
def HPyUnicode_AsUTF8String(space, ctx, h):
    w_unicode = handles.deref(space, h)
    # XXX: what should we do if w_unicode is not a str?
    w_bytes = unicodeobject.encode_object(space, w_unicode, 'utf-8', 'strict')
    return handles.new(space, w_bytes)

@API.func("HPy HPyUnicode_FromWideChar(HPyContext ctx, const wchar_t *w, HPy_ssize_t size)")
def HPyUnicode_FromWideChar(space, ctx, wchar_p, size):
    # remove the "const", else we can't call wcharpsize2utf8 later
    wchar_p = rffi.cast(rffi.CWCHARP, wchar_p)
    if wchar_p:
        if size == -1:
            size = wcharplen(wchar_p)
        # WRITE TEST: this automatically raises "character not in range", but
        # we don't have any test for it
        s = wcharpsize2utf8(space, wchar_p, size)
        w_obj = space.newutf8(s, size)
        return handles.new(space, w_obj)
    else:
        # cpyext returns an empty string, we need a test
        raise NotImplementedError("WRITE TEST")


def wcharplen(wchar_p):
    i = 0
    while ord(wchar_p[i]):
        i += 1
    return i
