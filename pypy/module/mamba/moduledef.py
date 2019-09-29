from pypy.interpreter.mixedmodule import MixedModule

class Module(MixedModule):

  interpleveldefs = {
    'Bits'  : 'bits.W_Bits',
    # 'concat': 'bits_helpers.concat',
  }

  appleveldefs = {
  }
