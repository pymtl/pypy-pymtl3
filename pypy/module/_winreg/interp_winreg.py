from rpython.rtyper.lltypesystem import rffi, lltype
from rpython.rlib import rwinreg, rwin32
from rpython.rlib.rarithmetic import r_uint, intmask
from rpython.rlib.buffer import ByteBuffer

from pypy.interpreter.baseobjspace import W_Root, BufferInterfaceNotFound
from pypy.interpreter.gateway import interp2app, unwrap_spec
from pypy.interpreter.typedef import TypeDef, GetSetProperty
from pypy.interpreter.error import OperationError, oefmt, wrap_oserror
from pypy.interpreter.unicodehelper import (
        str_decode_utf_16, utf8_encode_utf_16)
from pypy.module._codecs.interp_codecs import CodecState


def raiseWindowsError(space, errcode, context):
    message = rwin32.FormatErrorW(errcode)
    w_errcode = space.newint(errcode)
    w_t = space.newtuple([w_errcode, space.newtext(*message),
                          space.w_None, w_errcode])
    raise OperationError(space.w_WindowsError, w_t)


class W_HKEY(W_Root):
    def __init__(self, space, hkey):
        self.hkey = hkey
        self.space = space
        self.register_finalizer(space)

    def _finalize_(self):
        # ignore errors
        try:
            self.Close(self.space)
        except:
            pass

    def as_int(self):
        return rffi.cast(rffi.SIZE_T, self.hkey)

    def descr_bool(self, space):
        return space.newbool(self.as_int() != 0)

    def descr_handle_get(self, space):
        return space.newint(self.as_int())

    def descr_repr(self, space):
        return space.newtext("<PyHKEY:0x%x>" % (self.as_int(),))

    def descr_int(self, space):
        return space.newint(self.as_int())

    def descr__enter__(self, space):
        return self

    def descr__exit__(self, space, __args__):
        CloseKey(space, self)

    def Close(self, space):
        """key.Close() - Closes the underlying Windows handle.
If the handle is already closed, no error is raised."""
        CloseKey(space, self)

    def Detach(self, space):
        """int = key.Detach() - Detaches the Windows handle from the handle object.

The result is the value of the handle before it is detached.  If the
handle is already detached, this will return zero.

After calling this function, the handle is effectively invalidated,
but the handle is not closed.  You would call this function when you
need the underlying win32 handle to exist beyond the lifetime of the
handle object.
On 64 bit windows, the result of this function is a long integer"""
        key = self.as_int()
        self.hkey = rwin32.NULL_HANDLE
        return space.newint(key)


@unwrap_spec(key=int)
def new_HKEY(space, w_subtype, key):
    hkey = rffi.cast(rwinreg.HKEY, key)
    return W_HKEY(space, hkey)


descr_HKEY_new = interp2app(new_HKEY)


W_HKEY.typedef = TypeDef(
    "winreg.HKEYType",
    __doc__="""\
PyHKEY Object - A Python object, representing a win32 registry key.

This object wraps a Windows HKEY object, automatically closing it when
the object is destroyed.  To guarantee cleanup, you can call either
the Close() method on the PyHKEY, or the CloseKey() method.

All functions which accept a handle object also accept an integer -
however, use of the handle object is encouraged.

Functions:
Close() - Closes the underlying handle.
Detach() - Returns the integer Win32 handle, detaching it from the object

Properties:
handle - The integer Win32 handle.

Operations:
__bool__ - Handles with an open object return true, otherwise false.
__int__ - Converting a handle to an integer returns the Win32 handle.
__cmp__ - Handle objects are compared using the handle value.""",
    __new__=descr_HKEY_new,
    __repr__=interp2app(W_HKEY.descr_repr),
    __int__=interp2app(W_HKEY.descr_int),
    __bool__=interp2app(W_HKEY.descr_bool),
    __enter__=interp2app(W_HKEY.descr__enter__),
    __exit__=interp2app(W_HKEY.descr__exit__),
    handle=GetSetProperty(W_HKEY.descr_handle_get),
    Close=interp2app(W_HKEY.Close),
    Detach=interp2app(W_HKEY.Detach),
    )


def hkey_w(w_hkey, space):
    if space.is_w(w_hkey, space.w_None):
        raise oefmt(space.w_TypeError,
                    "None is not a valid HKEY in this context")
    elif isinstance(w_hkey, W_HKEY):
        return w_hkey.hkey
    elif space.isinstance_w(w_hkey, space.w_int):
        if space.is_true(space.lt(w_hkey, space.newint(0))):
            return rffi.cast(rwinreg.HKEY, space.int_w(w_hkey))
        return rffi.cast(rwinreg.HKEY, space.uint_w(w_hkey))
    else:
        raise oefmt(space.w_TypeError, "The object is not a PyHKEY object")


def CloseKey(space, w_hkey):
    """CloseKey(hkey) - Closes a previously opened registry key.

The hkey argument specifies a previously opened key.

Note that if the key is not closed using this method, it will be
closed when the hkey object is destroyed by Python."""
    hkey = hkey_w(w_hkey, space)
    if hkey:
        ret = rwinreg.RegCloseKey(hkey)
        if ret != 0:
            raiseWindowsError(space, ret, 'RegCloseKey')
    if isinstance(w_hkey, W_HKEY):
        space.interp_w(W_HKEY, w_hkey).hkey = rwin32.NULL_HANDLE


def FlushKey(space, w_hkey):
    """FlushKey(key) - Writes all the attributes of a key to the registry.

key is an already open key, or any one of the predefined HKEY_* constants.

It is not necessary to call RegFlushKey to change a key.
Registry changes are flushed to disk by the registry using its lazy flusher.
Registry changes are also flushed to disk at system shutdown.
Unlike CloseKey(), the FlushKey() method returns only when all the data has
been written to the registry.
An application should only call FlushKey() if it requires absolute certainty
that registry changes are on disk.
If you don't know whether a FlushKey() call is required, it probably isn't."""
    hkey = hkey_w(w_hkey, space)
    if hkey:
        ret = rwinreg.RegFlushKey(hkey)
        if ret != 0:
            raiseWindowsError(space, ret, 'RegFlushKey')


@unwrap_spec(subkey="unicode", filename="unicode")
def LoadKey(space, w_hkey, subkey, filename):
    """LoadKey(key, sub_key, file_name) - Creates a subkey under the specified key
and stores registration information from a specified file into that subkey.

key is an already open key, or any one of the predefined HKEY_* constants.
sub_key is a string that identifies the sub_key to load
file_name is the name of the file to load registry data from.
 This file must have been created with the SaveKey() function.
 Under the file allocation table (FAT) file system, the filename may not
have an extension.

A call to LoadKey() fails if the calling process does not have the
SE_RESTORE_PRIVILEGE privilege.

If key is a handle returned by ConnectRegistry(), then the path specified
in fileName is relative to the remote computer.

The docs imply key must be in the HKEY_USER or HKEY_LOCAL_MACHINE tree"""
    # XXX should filename use space.fsencode_w?
    hkey = hkey_w(w_hkey, space)
    with rffi.scoped_unicode2wcharp(subkey) as wide_subkey:
        c_subkey = rffi.cast(rffi.CWCHARP, wide_subkey)
        with rffi.scoped_unicode2wcharp(filename) as wide_filename:
            c_filename = rffi.cast(rffi.CWCHARP, wide_filename)
            ret = rwinreg.RegLoadKeyW(hkey, c_subkey, c_filename)
            if ret != 0:
                raiseWindowsError(space, ret, 'RegLoadKey')


@unwrap_spec(filename="unicode")
def SaveKey(space, w_hkey, filename):
    """
SaveKey(key, file_name) - Saves the specified key, and all its subkeys to the
specified file.

key is an already open key, or any one of the predefined HKEY_* constants.
file_name is the name of the file to save registry data to.
 This file cannot already exist. If this filename includes an extension,
 it cannot be used on file allocation table (FAT) file systems by the
 LoadKey(), ReplaceKey() or RestoreKey() methods.

If key represents a key on a remote computer, the path described by
file_name is relative to the remote computer.
The caller of this method must possess the SeBackupPrivilege security
privilege. This function passes NULL for security_attributes to the API."""
    hkey = hkey_w(w_hkey, space)
    with rffi.scoped_unicode2wcharp(filename) as wide_filename:
        c_filename = rffi.cast(rffi.CWCHARP, wide_filename)
        ret = rwinreg.RegSaveKeyW(hkey, c_filename, None)
        if ret != 0:
            raiseWindowsError(space, ret, 'RegSaveKey')


@unwrap_spec(typ=int)
def SetValue(space, w_hkey, w_subkey, typ, w_value):
    """
SetValue(key, sub_key, type, value) - Associates a value with a specified key.

key is an already open key, or any one of the predefined HKEY_* constants.
sub_key is a string that names the subkey with which the value is associated.
type is an integer that specifies the type of the data.  Currently this
 must be REG_SZ, meaning only strings are supported.
value is a string that specifies the new value.

If the key specified by the sub_key parameter does not exist, the SetValue
function creates it.

Value lengths are limited by available memory. Long values (more than
2048 bytes) should be stored as files with the filenames stored in
the configuration registry.  This helps the registry perform efficiently.

The key identified by the key parameter must have been opened with
KEY_SET_VALUE access."""
    if typ != rwinreg.REG_SZ:
        raise oefmt(space.w_ValueError, "Type must be winreg.REG_SZ")
    hkey = hkey_w(w_hkey, space)
    state = space.fromcache(CodecState)
    errh = state.encode_error_handler
    utf8 = space.utf8_w(w_subkey)
    subkeyW = utf8_encode_utf_16(utf8 + '\x00', 'strict', errh, allow_surrogates=False)
    utf8 = space.utf8_w(w_value)
    valueW = utf8_encode_utf_16(utf8 + '\x00', 'strict', errh, allow_surrogates=False)
    valueL = space.len_w(w_value)

    # Add an offset to remove the BOM from the native utf16 wstr
    with rffi.scoped_nonmovingbuffer(subkeyW) as subkeyP0:
        subkeyP = rffi.cast(rffi.CWCHARP, rffi.ptradd(subkeyP0, 2))
        with rffi.scoped_nonmovingbuffer(valueW) as valueP0:
            valueP = rffi.cast(rffi.CWCHARP, rffi.ptradd(valueP0, 2))
            ret = rwinreg.RegSetValueW(hkey, subkeyP, rwinreg.REG_SZ,
                                       valueP, valueL)
            if ret != 0:
                raiseWindowsError(space, ret, 'RegSetValue')


def QueryValue(space, w_hkey, w_subkey):
    """
string = QueryValue(key, sub_key) - retrieves the unnamed value for a key.

key is an already open key, or any one of the predefined HKEY_* constants.
sub_key is a string that holds the name of the subkey with which the value
 is associated.  If this parameter is None or empty, the function retrieves
 the value set by the SetValue() method for the key identified by key.

Values in the registry have name, type, and data components. This method
retrieves the data for a key's first value that has a NULL name.
But the underlying API call doesn't return the type: Lame, DONT USE THIS!!!"""
    hkey = hkey_w(w_hkey, space)
    if space.is_w(w_subkey, space.w_None):
        subkey = None
    else:
        subkey = space.utf8_w(w_subkey).decode('utf8')
    with rffi.scoped_unicode2wcharp(subkey) as wide_subkey:
        c_subkey = rffi.cast(rffi.CWCHARP, wide_subkey)
        with lltype.scoped_alloc(rwin32.PLONG.TO, 1) as bufsize_p:
            bufsize_p[0] = 0
            ret = rwinreg.RegQueryValueW(hkey, c_subkey, None, bufsize_p)
            if ret == 0 and intmask(bufsize_p[0]) == 0:
                return space.newtext('', 0)
            elif ret != 0 and ret != rwinreg.ERROR_MORE_DATA:
                raiseWindowsError(space, ret, 'RegQueryValue')
            # Add extra space for a NULL ending
            buf = ByteBuffer(intmask(bufsize_p[0]) * 2 + 2)
            bufP = rffi.cast(rwin32.LPWSTR, buf.get_raw_address())
            ret = rwinreg.RegQueryValueW(hkey, c_subkey, bufP, bufsize_p)
            if ret != 0:
                raiseWindowsError(space, ret, 'RegQueryValue')
            utf8, lgt = wbuf_to_utf8(space, buf[0:intmask(bufsize_p[0])])
            return space.newtext(utf8, lgt)


def convert_to_regdata(space, w_value, typ):
    '''returns CCHARP, int'''
    buf = None

    if typ == rwinreg.REG_DWORD:
        if space.is_none(w_value) or space.isinstance_w(w_value, space.w_int):
            if space.is_none(w_value):
                value = r_uint(0)
            else:
                value = space.c_uint_w(w_value)
            buflen = rffi.sizeof(rwin32.DWORD)
            buf1 = lltype.malloc(rffi.CArray(rwin32.DWORD), 1, flavor='raw')
            buf1[0] = value
            buf = rffi.cast(rffi.CCHARP, buf1)

    elif typ == rwinreg.REG_SZ or typ == rwinreg.REG_EXPAND_SZ:
        if space.is_w(w_value, space.w_None):
            buflen = 1
            buf = lltype.malloc(rffi.CCHARP.TO, buflen, flavor='raw')
            buf[0] = '\0'
        else:
            buf = rffi.unicode2wcharp(space.utf8_w(w_value).decode('utf8'))
            buf = rffi.cast(rffi.CCHARP, buf)
            buflen = (space.len_w(w_value) * 2) + 1

    elif typ == rwinreg.REG_MULTI_SZ:
        if space.is_w(w_value, space.w_None):
            buflen = 1
            buf = lltype.malloc(rffi.CCHARP.TO, buflen, flavor='raw')
            buf[0] = '\0'
        elif space.isinstance_w(w_value, space.w_list):
            strings = []
            buflen = 0

            # unwrap strings and compute total size
            w_iter = space.iter(w_value)
            while True:
                try:
                    w_item = space.next(w_iter)
                    item = space.utf8_w(w_item).decode('utf8')
                    strings.append(item)
                    buflen += 2 * (len(item) + 1)
                except OperationError as e:
                    if not e.match(space, space.w_StopIteration):
                        raise       # re-raise other app-level exceptions
                    break
            buflen += 2
            buf = lltype.malloc(rffi.CCHARP.TO, buflen, flavor='raw')

            # Now copy data
            buflen = 0
            for string in strings:
                with rffi.scoped_unicode2wcharp(string) as wchr:
                    c_str = rffi.cast(rffi.CCHARP, wchr)
                    for i in range(len(string) * 2):
                        buf[buflen + i] = c_str[i]
                buflen += (len(string) + 1) * 2
                buf[buflen - 1] = '\0'
                buf[buflen - 2] = '\0'
            buflen += 2
            buf[buflen - 1] = '\0'
            buf[buflen - 2] = '\0'

    else:  # REG_BINARY and ALL unknown data types.
        if space.is_w(w_value, space.w_None):
            buflen = 0
            buf = lltype.malloc(rffi.CCHARP.TO, 1, flavor='raw')
            buf[0] = '\0'
        else:
            try:
                value = w_value.buffer_w(space, space.BUF_SIMPLE)
            except BufferInterfaceNotFound:
                raise oefmt(space.w_TypeError,
                            "Objects of type '%T' can not be used as binary "
                            "registry values", w_value)
            else:
                value = value.as_str()
            buflen = len(value)
            buf = rffi.str2charp(value)

    if buf is not None:
        return rffi.cast(rffi.CWCHARP, buf), buflen

    raise oefmt(space.w_ValueError,
                "Could not convert the data to the specified type")


def wbuf_to_utf8(space, wbuf):
    state = space.fromcache(CodecState)
    errh = state.decode_error_handler
    utf8, lgt, pos = str_decode_utf_16(wbuf, 'strict', final=True,
                                       errorhandler=errh)
    if len(utf8) > 1 and utf8[len(utf8) - 1] == '\x00':
        # trim off one trailing '\x00'
        newlen = len(utf8) - 1
        assert newlen >=0
        utf8 = utf8[0:newlen]
        lgt -= 1
    return utf8, lgt


def convert_from_regdata(space, buf, buflen, typ):
    if typ == rwinreg.REG_DWORD:
        if not buflen:
            return space.newint(0)
        d = rffi.cast(rwin32.LPDWORD, buf.get_raw_address())[0]
        return space.newint(d)

    elif typ == rwinreg.REG_SZ or typ == rwinreg.REG_EXPAND_SZ:
        if not buflen:
            return space.newtext('', 0)
        even = (buflen // 2) * 2
        utf8, lgt = wbuf_to_utf8(space, buf[0:even])
        # may have a trailing NULL in the buffer.
        utf8len = len(utf8)
        if utf8len > 2 and lgt > 0 and utf8[utf8len - 2] == '\x00':
            lgt -= 1
            newlen = len(utf8) - 1
            assert newlen > 0
            utf8 = utf8[0:newlen]
        w_s = space.newtext(utf8, lgt)
        return w_s

    elif typ == rwinreg.REG_MULTI_SZ:
        if not buflen:
            return space.newlist([])
        i = 0
        ret = []
        while i < buflen and buf[i]:
            start = i
            while i < buflen and buf[i] != '\0':
                i += 2
            if start == i:
                break
            utf8, lgt = wbuf_to_utf8(space, buf[start:i])
            ret.append(space.newtext(utf8, lgt))
            i += 2
        return space.newlist(ret)

    else:  # REG_BINARY and all other types
        return space.newbytes(buf[0:buflen])


@unwrap_spec(value_name="unicode", typ=int)
def SetValueEx(space, w_hkey, value_name, w_reserved, typ, w_value):
    """
SetValueEx(key, value_name, reserved, type, value) - Stores data in the value
field of an open registry key.

key is an already open key, or any one of the predefined HKEY_* constants.
value_name is a string containing the name of the value to set, or None
type is an integer that specifies the type of the data.  This should be one of:
  REG_BINARY -- Binary data in any form.
  REG_DWORD -- A 32-bit number.
  REG_DWORD_LITTLE_ENDIAN -- A 32-bit number in little-endian format.
  REG_DWORD_BIG_ENDIAN -- A 32-bit number in big-endian format.
  REG_EXPAND_SZ -- A null-terminated string that contains unexpanded references
                   to environment variables (for example, %PATH%).
  REG_LINK -- A Unicode symbolic link.
  REG_MULTI_SZ -- An sequence of null-terminated strings, terminated by
                  two null characters.  Note that Python handles this
                  termination automatically.
  REG_NONE -- No defined value type.
  REG_RESOURCE_LIST -- A device-driver resource list.
  REG_SZ -- A null-terminated string.
reserved can be anything - zero is always passed to the API.
value is a string that specifies the new value.

This method can also set additional value and type information for the
specified key.  The key identified by the key parameter must have been
opened with KEY_SET_VALUE access.

To open the key, use the CreateKeyEx() or OpenKeyEx() methods.

Value lengths are limited by available memory. Long values (more than
2048 bytes) should be stored as files with the filenames stored in
the configuration registry.  This helps the registry perform efficiently."""
    hkey = hkey_w(w_hkey, space)
    buf, buflen = convert_to_regdata(space, w_value, typ)
    try:
        with rffi.scoped_unicode2wcharp(value_name) as wide_vn:
            c_vn = rffi.cast(rffi.CWCHARP, wide_vn)
            ret = rwinreg.RegSetValueExW(hkey, c_vn, 0, typ, buf, buflen)
    finally:
        lltype.free(buf, flavor='raw')
    if ret != 0:
        raiseWindowsError(space, ret, 'RegSetValueEx')


def QueryValueEx(space, w_hkey, w_subkey):
    """
value,type_id = QueryValueEx(key, value_name) - Retrieves the type and data for
a specified value name associated with an open registry key.

key is an already open key, or any one of the predefined HKEY_* constants.
value_name is a string indicating the value to query"""
    hkey = hkey_w(w_hkey, space)
    if space.is_w(w_subkey, space.w_None):
        subkey = None
    else:
        subkey = space.utf8_w(w_subkey).decode('utf8')
    null_dword = lltype.nullptr(rwin32.LPDWORD.TO)
    with rffi.scoped_unicode2wcharp(subkey) as wide_subkey:
        c_subkey = rffi.cast(rffi.CWCHARP, wide_subkey)
        with lltype.scoped_alloc(rwin32.LPDWORD.TO, 1) as dataSize:
            ret = rwinreg.RegQueryValueExW(hkey, c_subkey, null_dword,
                                           null_dword, None, dataSize)
            bufSize = intmask(dataSize[0])
            if ret == rwinreg.ERROR_MORE_DATA:
                # Copy CPython behaviour, otherwise bufSize can be 0
                bufSize = 256
            elif ret != 0:
                raiseWindowsError(space, ret, 'RegQueryValue')
            while True:
                dataBuf = ByteBuffer(bufSize)
                dataBufP = rffi.cast(rffi.CWCHARP, dataBuf.get_raw_address())
                with lltype.scoped_alloc(rwin32.LPDWORD.TO, 1) as retType:

                    ret = rwinreg.RegQueryValueExW(hkey, c_subkey, null_dword,
                                                   retType, dataBufP, dataSize)
                    if ret == rwinreg.ERROR_MORE_DATA:
                        # Resize and retry
                        bufSize *= 2
                        dataSize[0] = rffi.cast(rwin32.DWORD, bufSize)
                        continue
                    if ret != 0:
                        raiseWindowsError(space, ret, 'RegQueryValueEx')
                    length = intmask(dataSize[0])
                    return space.newtuple([
                        convert_from_regdata(space, dataBuf,
                                             length, retType[0]),
                        space.newint(intmask(retType[0])),
                    ])


@unwrap_spec(subkey="unicode")
def CreateKey(space, w_hkey, subkey):
    """key = CreateKey(key, sub_key) - Creates or opens the specified key.

key is an already open key, or one of the predefined HKEY_* constants
sub_key is a string that names the key this method opens or creates.
 If key is one of the predefined keys, sub_key may be None. In that case,
 the handle returned is the same key handle passed in to the function.

If the key already exists, this function opens the existing key

The return value is the handle of the opened key.
If the function fails, an exception is raised."""
    hkey = hkey_w(w_hkey, space)
    with rffi.scoped_unicode2wcharp(subkey) as wide_subkey:
        c_subkey = rffi.cast(rffi.CWCHARP, wide_subkey)
        with lltype.scoped_alloc(rwinreg.PHKEY.TO, 1) as rethkey:
            ret = rwinreg.RegCreateKeyW(hkey, c_subkey, rethkey)
            if ret != 0:
                raiseWindowsError(space, ret, 'CreateKey')
            return W_HKEY(space, rethkey[0])


@unwrap_spec(sub_key="unicode", reserved=int, access=rffi.r_uint)
def CreateKeyEx(space, w_key, sub_key, reserved=0, access=rwinreg.KEY_WRITE):
    """key = CreateKey(key, sub_key) - Creates or opens the specified key.

key is an already open key, or one of the predefined HKEY_* constants
sub_key is a string that names the key this method opens or creates.
 If key is one of the predefined keys, sub_key may be None. In that case,
 the handle returned is the same key handle passed in to the function.

If the key already exists, this function opens the existing key

The return value is the handle of the opened key.
If the function fails, an exception is raised."""
    hkey = hkey_w(w_key, space)
    with rffi.scoped_unicode2wcharp(sub_key) as wide_sub_key:
        c_subkey = rffi.cast(rffi.CWCHARP, wide_sub_key)
        with lltype.scoped_alloc(rwinreg.PHKEY.TO, 1) as rethkey:
            ret = rwinreg.RegCreateKeyExW(hkey, c_subkey, reserved, None, 0,
                                          access, None, rethkey,
                                          lltype.nullptr(rwin32.LPDWORD.TO))
            if ret != 0:
                raiseWindowsError(space, ret, 'CreateKeyEx')
            return W_HKEY(space, rethkey[0])


@unwrap_spec(subkey="unicode")
def DeleteKey(space, w_hkey, subkey):
    """
DeleteKey(key, subkey) - Deletes the specified key.

key is an already open key, or any one of the predefined HKEY_* constants.
sub_key is a string that must be a subkey of the key identified by the key
parameter. This value must not be None, and the key may not have subkeys.

This method can not delete keys with subkeys.

If the method succeeds, the entire key, including all of its values,
is removed.  If the method fails, an EnvironmentError exception is raised."""
    hkey = hkey_w(w_hkey, space)
    with rffi.scoped_unicode2wcharp(subkey) as wide_subkey:
        c_subkey = rffi.cast(rffi.CWCHARP, wide_subkey)
        ret = rwinreg.RegDeleteKeyW(hkey, c_subkey)
        if ret != 0:
            raiseWindowsError(space, ret, 'RegDeleteKey')


@unwrap_spec(subkey="unicode")
def DeleteValue(space, w_hkey, subkey):
    """DeleteValue(key, value) - Removes a named value from a registry key.

key is an already open key, or any one of the predefined HKEY_* constants.
value is a string that identifies the value to remove."""
    hkey = hkey_w(w_hkey, space)
    with rffi.scoped_unicode2wcharp(subkey) as wide_subkey:
        c_subkey = rffi.cast(rffi.CWCHARP, wide_subkey)
        ret = rwinreg.RegDeleteValueW(hkey, c_subkey)
        if ret != 0:
            raiseWindowsError(space, ret, 'RegDeleteValue')


@unwrap_spec(reserved=int, access=rffi.r_uint)
def OpenKey(space, w_key, w_sub_key, reserved=0, access=rwinreg.KEY_READ):
    """
key = OpenKey(key, sub_key, res = 0, sam = KEY_READ) - Opens the specified key.

key is an already open key, or any one of the predefined HKEY_* constants.
sub_key is a string that identifies the sub_key to open
res is a reserved integer, and must be zero.  Default is zero.
sam is an integer that specifies an access mask that describes the desired
 security access for the key.  Default is KEY_READ

The result is a new handle to the specified key
If the function fails, an EnvironmentError exception is raised."""
    hkey = hkey_w(w_key, space)
    utf8 = space.utf8_w(w_sub_key)
    state = space.fromcache(CodecState)
    errh = state.encode_error_handler
    subkeyW = utf8_encode_utf_16(utf8 + '\x00', 'strict', errh, allow_surrogates=False)
    with rffi.scoped_nonmovingbuffer(subkeyW) as subkeyP0:
        subkeyP = rffi.cast(rffi.CWCHARP, rffi.ptradd(subkeyP0, 2))
        with lltype.scoped_alloc(rwinreg.PHKEY.TO, 1) as rethkey:
            ret = rwinreg.RegOpenKeyExW(hkey, subkeyP, reserved, access,
                                        rethkey)
            if ret != 0:
                raiseWindowsError(space, ret, 'RegOpenKeyEx')
            return W_HKEY(space, rethkey[0])


@unwrap_spec(index=int)
def EnumValue(space, w_hkey, index):
    """tuple = EnumValue(key, index) - Enumerates values of an open registry key.
key is an already open key, or any one of the predefined HKEY_* constants.
index is an integer that identifies the index of the value to retrieve.

The function retrieves the name of one subkey each time it is called.
It is typically called repeatedly, until an EnvironmentError exception
is raised, indicating no more values.

The result is a tuple of 3 items:
value_name is a string that identifies the value.
value_data is an object that holds the value data, and whose type depends
 on the underlying registry type.
data_type is an integer that identifies the type of the value data."""
    hkey = hkey_w(w_hkey, space)
    null_dword = lltype.nullptr(rwin32.LPDWORD.TO)

    with lltype.scoped_alloc(rwin32.LPDWORD.TO, 1) as valueSize:
        with lltype.scoped_alloc(rwin32.LPDWORD.TO, 1) as dataSize:
            ret = rwinreg.RegQueryInfoKeyW(
                hkey, None, null_dword, null_dword,
                null_dword, null_dword, null_dword,
                null_dword, valueSize, dataSize,
                null_dword, lltype.nullptr(rwin32.PFILETIME.TO))
            if ret != 0:
                raiseWindowsError(space, ret, 'RegQueryInfoKey')
            # include null terminators
            valueSize[0] += 1
            dataSize[0] += 1
            bufDataSize = intmask(dataSize[0])
            bufValueSize = intmask(valueSize[0] * 2)

            valueBuf = ByteBuffer(bufValueSize)
            valueBufP = rffi.cast(rffi.CWCHARP, valueBuf.get_raw_address())
            while True:
                dataBuf = ByteBuffer(bufDataSize)
                dataBufP = rffi.cast(rffi.CCHARP, dataBuf.get_raw_address())
                with lltype.scoped_alloc(rwin32.LPDWORD.TO,
                                         1) as retType:
                    ret = rwinreg.RegEnumValueW(
                        hkey, index, valueBufP, valueSize,
                        null_dword, retType, dataBufP, dataSize)
                    if ret == rwinreg.ERROR_MORE_DATA:
                        # Resize and retry. For dynamic keys, the value of
                        # dataSize[0] is useless (always 1) so do what CPython
                        # does, except they use 2 instead of 4
                        bufDataSize *= 4
                        dataSize[0] = rffi.cast(rwin32.DWORD,
                                                bufDataSize)
                        valueSize[0] = rffi.cast(rwin32.DWORD,
                                                 bufValueSize)
                        continue

                    if ret != 0:
                        raiseWindowsError(space, ret, 'RegEnumValue')

                    length = intmask(dataSize[0])
                    vlen = (intmask(valueSize[0]) + 1) * 2
                    utf8v, lenv = wbuf_to_utf8(space, valueBuf[0:vlen])
                    return space.newtuple([
                        space.newtext(utf8v, lenv),
                        convert_from_regdata(space, dataBuf,
                                             length, retType[0]),
                        space.newint(intmask(retType[0])),
                        ])


@unwrap_spec(index=int)
def EnumKey(space, w_hkey, index):
    """string = EnumKey(key, index) - Enumerates subkeys of an open registry key.

key is an already open key, or any one of the predefined HKEY_* constants.
index is an integer that identifies the index of the key to retrieve.

The function retrieves the name of one subkey each time it is called.
It is typically called repeatedly until an EnvironmentError exception is
raised, indicating no more values are available."""
    hkey = hkey_w(w_hkey, space)
    null_dword = lltype.nullptr(rwin32.LPDWORD.TO)

    # The Windows docs claim that the max key name length is 255
    # characters, plus a terminating nul character.  However,
    # empirical testing demonstrates that it is possible to
    # create a 256 character key that is missing the terminating
    # nul.  RegEnumKeyEx requires a 257 character buffer to
    # retrieve such a key name.
    buf = ByteBuffer(257 * 2)
    bufP = rffi.cast(rwin32.LPWSTR, buf.get_raw_address())
    with lltype.scoped_alloc(rwin32.LPDWORD.TO, 1) as valueSize:
        valueSize[0] = r_uint(257)  # includes NULL terminator
        ret = rwinreg.RegEnumKeyExW(hkey, index, bufP, valueSize,
                                    null_dword, None, null_dword,
                                    lltype.nullptr(rwin32.PFILETIME.TO))
        if ret != 0:
            raiseWindowsError(space, ret, 'RegEnumKeyEx')
        vlen = intmask(valueSize[0]) * 2
        utf8, lgt = wbuf_to_utf8(space, buf[0:vlen])
        return space.newtext(utf8, lgt)


def QueryInfoKey(space, w_hkey):
    """tuple = QueryInfoKey(key) - Returns information about a key.

key is an already open key, or any one of the predefined HKEY_* constants.

The result is a tuple of 3 items:
An integer that identifies the number of sub keys this key has.
An integer that identifies the number of values this key has.
A long integer that identifies when the key was last modified (if available)
 as 100's of nanoseconds since Jan 1, 1600."""
    hkey = hkey_w(w_hkey, space)
    with lltype.scoped_alloc(rwin32.LPDWORD.TO, 1) as nSubKeys:
        with lltype.scoped_alloc(rwin32.LPDWORD.TO, 1) as nValues:
            with lltype.scoped_alloc(rwin32.PFILETIME.TO, 1) as ft:
                null_dword = lltype.nullptr(rwin32.LPDWORD.TO)
                ret = rwinreg.RegQueryInfoKeyW(
                    hkey, None, null_dword, null_dword,
                    nSubKeys, null_dword, null_dword,
                    nValues, null_dword, null_dword,
                    null_dword, ft)
                if ret != 0:
                    raiseWindowsError(space, ret, 'RegQueryInfoKey')
                lgt = ((lltype.r_longlong(ft[0].c_dwHighDateTime) << 32) +
                       lltype.r_longlong(ft[0].c_dwLowDateTime))
                return space.newtuple([space.newint(nSubKeys[0]),
                                       space.newint(nValues[0]),
                                       space.newint(lgt)])


def ConnectRegistry(space, w_machine, w_hkey):
    """
key = ConnectRegistry(computer_name, key)

Establishes a connection to a predefined registry handle on another computer.

computer_name is the name of the remote computer, of the form \\\\computername.
 If None, the local computer is used.
key is the predefined handle to connect to.

The return value is the handle of the opened key.
If the function fails, an EnvironmentError exception is raised."""
    hkey = hkey_w(w_hkey, space)
    if space.is_none(w_machine):
        with lltype.scoped_alloc(rwinreg.PHKEY.TO, 1) as rethkey:
            ret = rwinreg.RegConnectRegistryW(None, hkey, rethkey)
            if ret != 0:
                raiseWindowsError(space, ret, 'RegConnectRegistry')
            return W_HKEY(space, rethkey[0])
    else:
        utf8 = space.utf8_w(w_machine)
        state = space.fromcache(CodecState)
        errh = state.encode_error_handler
        machineW = utf8_encode_utf_16(utf8 + '\x00', 'strict', errh, allow_surrogates=False)
        with rffi.scoped_nonmovingbuffer(machineW) as machineP0:
            machineP = rffi.cast(rwin32.LPWSTR, rffi.ptradd(machineP0, 2))
            with lltype.scoped_alloc(rwinreg.PHKEY.TO, 1) as rethkey:
                ret = rwinreg.RegConnectRegistryW(machineP, hkey, rethkey)
            if ret != 0:
                raiseWindowsError(space, ret, 'RegConnectRegistry')
            return W_HKEY(space, rethkey[0])


def ExpandEnvironmentStrings(space, w_source):
    "string = ExpandEnvironmentStrings(string) - Expand environment vars."
    try:
        source, source_ulen = space.utf8_len_w(w_source)
        res, res_ulen = rwinreg.ExpandEnvironmentStrings(source, source_ulen)
        return space.newutf8(res, res_ulen)
    except WindowsError as e:
        raise wrap_oserror(space, e)


def DisableReflectionKey(space, w_key):
    """Disables registry reflection for 32-bit processes running on a 64-bit
    Operating System.  Will generally raise NotImplemented if executed on
    a 32-bit Operating System.
    If the key is not on the reflection list, the function succeeds but has no
    effect.  Disabling reflection for a key does not affect reflection of any
    subkeys."""
    raise oefmt(space.w_NotImplementedError,
                "not implemented on this platform")


def EnableReflectionKey(space, w_key):
    """Restores registry reflection for the specified disabled key.
    Will generally raise NotImplemented if executed on a 32-bit Operating
    System.  Restoring reflection for a key does not affect reflection of any
    subkeys."""
    raise oefmt(space.w_NotImplementedError,
                "not implemented on this platform")


def QueryReflectionKey(space, w_key):
    """bool = QueryReflectionKey(hkey) - Determines the reflection state for
    the specified key.  Will generally raise NotImplemented if executed on a
    32-bit Operating System."""
    raise oefmt(space.w_NotImplementedError,
                "not implemented on this platform")


@unwrap_spec(sub_key="unicode", reserved=int, access=rffi.r_uint)
def DeleteKeyEx(space, w_key, sub_key, reserved=0, access=rwinreg.KEY_WOW64_64KEY):
    """DeleteKeyEx(key, sub_key, sam, res) - Deletes the specified key.

    key is an already open key, or any one of the predefined HKEY_* constants.
    sub_key is a string that must be a subkey of the key identified by the key
    parameter.
    res is a reserved integer, and must be zero.  Default is zero.
    sam is an integer that specifies an access mask that describes the desired
    This value must not be None, and the key may not have subkeys.

    This method can not delete keys with subkeys.

    If the method succeeds, the entire key, including all of its values,
    is removed.  If the method fails, a WindowsError exception is raised.
    On unsupported Windows versions, NotImplementedError is raised."""
    raise oefmt(space.w_NotImplementedError,
                "not implemented on this platform")
