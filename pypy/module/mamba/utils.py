from pypy.interpreter.error import oefmt
from pypy.interpreter.pycode import PyCode
from pypy.interpreter.pyopcode import source_as_str

def exec_pymtl( space, w_prog, w_globals=None, w_locals=None ):
  frame = space.getexecutioncontext().gettopframe()
  ec = space.getexecutioncontext()
  flags = ec.compiler.getcodeflags(frame.pycode)

  if space.isinstance_w(w_prog, space.gettypeobject(PyCode.typedef)):
    code = space.interp_w(PyCode, w_prog)
  else:
    from pypy.interpreter.astcompiler import consts
    flags |= consts.PyCF_SOURCE_IS_UTF8
    source, flags = source_as_str(space, w_prog, 'exec',
                                  "string, bytes or code", flags)
    code = ec.compiler.compile(source, "<string>", 'exec', flags)

  if (not space.is_none(w_globals) and not space.isinstance_w(w_globals, space.w_dict)):
    raise oefmt(space.w_TypeError, 'exec() arg 2 must be a dict, not %T', w_globals)
  if (not space.is_none(w_locals) and space.lookup(w_locals, '__getitem__') is None):
    raise oefmt(space.w_TypeError, 'exec() arg 3 must be a mapping or None, not %T', w_locals)

  if space.is_none(w_globals):
    w_globals = frame.get_w_globals()
    if space.is_none(w_locals):
      w_locals = frame.getdictscope()

  elif space.is_none(w_locals):
    w_locals = w_globals

  # Shunning: This is from EXEC_STMT in pypy for python 2..
  w_prog, w_globals, w_locals = space.fixedview( space.newtuple( [w_prog, w_globals, w_locals] ), 3 )

  space.call_method(w_globals, 'setdefault', space.newtext('__builtins__'),
                    frame.get_builtin())

  plain = (frame.get_w_locals() is not None and
           space.is_w(w_locals, frame.get_w_locals()))
  if plain:
    w_locals = frame.getdictscope()
  code.exec_code(space, w_globals, w_locals)
  if plain:
    frame.setdictscope(w_locals)
