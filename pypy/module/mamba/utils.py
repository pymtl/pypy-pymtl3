from pypy.interpreter import gateway

def exec_pymtl(space, w_prog, w_globals=None, w_locals=None):
    frame = space.getexecutioncontext().gettopframe()
    w_globals = normalize_dict(space, w_globals)
    w_locals = normalize_dict(space, w_locals)
    frame.exec_(w_prog, w_globals, w_locals)

app = gateway.applevel(r'''
    def normalize_dict( _dict ):

      # return the original dictionary if not running on
      # PyPy
      try:
        from __pypy__ import strategy, newdict
      except:
        return _dict

      # return the original dictionary if already using
      # ModuleDictStrategy
      if strategy( _dict ) == "ModuleDictStrategy":
        return _dict

      # create a new module dict
      new_dict = newdict("module")

      # copy over entries
      for key,value in _dict.items():
        new_dict[key] = value

      return new_dict
''', filename=__file__)

normalize_dict = app.interphook('normalize_dict')
