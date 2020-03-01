from pypy.interpreter.mixedmodule import MixedModule

class Module(MixedModule):

  interpleveldefs = {
    'Bits'  : 'smallbits.W_AbstractBits',
    'concat': 'utils.concat',
    'read_bytearray_bits': 'utils.read_bytearray_bits',
  }

  appleveldefs = {
  }
