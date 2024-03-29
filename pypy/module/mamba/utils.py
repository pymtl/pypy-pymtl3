
from rpython.rlib.rbigint import rbigint, SHIFT, _store_digit

from pypy.interpreter import gateway
from pypy.interpreter.error import OperationError, oefmt
from pypy.objspace.std.bytearrayobject import W_BytearrayObject
from pypy.objspace.std.intobject import W_IntObject
from pypy.objspace.std.longobject import W_LongObject

from pypy.module.mamba.smallbits import W_AbstractBits, W_SmallBits, get_int_mask
from pypy.module.mamba.bigbits   import W_BigBits, setitem_long_int_helper, setitem_long_long_helper

# @jit.look_inside_iff(lambda space, args_w:
        # jit.loop_unrolling_heuristic(args_w, len(args_w), 3))
    # case of multiple arguments (at least two).  We unroll it if there
    # are 2 or 3 arguments.
def concat_impl(space, args):
  args_w = args.arguments_w
  num_args = len(args_w)
  if num_args == 1:
    return args_w[0]

  nbits = 0
  for i in range(num_args):
    arg_w = args_w[i]
    if isinstance( arg_w, W_AbstractBits ):
      nbits += arg_w.nbits
    else:
      raise oefmt(space.w_TypeError,
                  "%d-th argument is of wrong type. Concat only takes Bits objects.",
                  i)

  stop = nbits

  if nbits <= SHIFT: # arg_w.nbits must <= SHIFT
    intval = 0

    for i in range(num_args):
      arg_w = args_w[i]
      assert isinstance( arg_w, W_SmallBits )

      slice_nbits = arg_w.nbits
      start = stop - slice_nbits
      valuemask  = ~(get_int_mask(slice_nbits) << start)
      intval = (intval & valuemask) | (arg_w.intval << start)
      stop = start

    return W_SmallBits( nbits, intval )

  else:
    # ret > SHIFT-bits, need to have rbigint
    bigval = rbigint.fromint(0)

    for i in range(num_args):
      arg_w = args_w[i]

      if isinstance( arg_w, W_SmallBits ):
        start = stop - arg_w.nbits
        bigval = setitem_long_int_helper( bigval, arg_w.intval, start, stop )
        stop = start
      elif isinstance( arg_w, W_BigBits ):
        start = stop - arg_w.nbits
        bigval = setitem_long_long_helper( bigval, arg_w.bigval, start, stop )
        stop = start

    # digits = []
    # curword = 0
    # start = 0

    # i = num_args - 1
    # while i >= 0:
      # arg_w = args_w[i]
      # i -= 1

      # if isinstance( arg_w, W_SmallBits ):
        # item = arg_w.intval

        # stop = start + arg_w.nbits # put arg_w into [start:stop]
        # if stop < SHIFT: # we are good
          # curword |= item << start
          # start = stop
        # else:
          # this_nbits = SHIFT - start

          # curword |= (item & get_int_mask(this_nbits) ) << start
          # digits.append(_store_digit(curword))

          # curword = item >> this_nbits
          # start = stop - SHIFT

      # elif isinstance( arg_w, W_BigBits ):
        # item = arg_w.bigval

        # nbits = arg_w.nbits
        # stop = start + nbits # put arg_w into [start:stop]

        # numwords = (nbits-1) / SHIFT
        # rembits  = arg_w.nbits - num_words * SHIFT
        # numwords += 1

        # if start <= SHIFT:
          # for i in range(numwords):
    return W_BigBits( nbits, bigval )

def concat(space, __args__):
  """concat( v1, v2, v3, ... )"""
  return concat_impl( space, __args__ )

# read_bytearray_bits

def read_bytearray_bits_impl( space, w_arr, w_addr, w_nbytes ):
  # We directly manipulate bytearray here
  assert isinstance( w_arr, W_BytearrayObject )
  ba_data   = w_arr._data
  ba_len    = len(ba_data)
  ba_offset = w_arr._offset

  nbytes = 0
  if   isinstance(w_nbytes, W_SmallBits):
    nbytes = w_nbytes.intval
  elif type(w_nbytes) is W_IntObject:
    nbytes = w_nbytes.intval
  elif isinstance(w_nbytes, W_BigBits):
    tmp = w_nbytes.bigval
    if tmp.numdigits() > 1:
      raise oefmt(space.w_ValueError, "nbytes [%s] too big for bytearray read Bits%d",
                                      rbigint.str(tmp), w_nbytes.nbits )
    nbytes = tmp.digit(0)
  elif type(w_nbytes) is W_LongObject:
    nbytes = w_nbytes.num.toint()
  else:
    raise oefmt(space.w_TypeError, "Please pass in int/Bits" )

  addr = 0
  if   isinstance(w_addr, W_SmallBits):
    addr = w_addr.intval
  elif type(w_addr) is W_IntObject:
    addr = w_addr.intval
  elif isinstance(w_addr, W_BigBits):
    tmp = w_addr.bigval
    if tmp.numdigits() > 1:
      raise oefmt(space.w_ValueError, "Index [%s] too big for bytearray read Bits%d",
                                      rbigint.str(tmp), w_addr.nbits )
    addr = tmp.digit(0)
  elif type(w_addr) is W_LongObject:
    addr = w_addr.num.toint()
  else:
    raise oefmt(space.w_TypeError, "Please pass in int/Bits" )

  if addr < 0:
    raise oefmt(space.w_ValueError, "read_bytearray_bits only accept positive addr.")

  begin = addr + ba_offset
  end   = begin + nbytes

  if end > ba_len:
    raise OperationError(space.w_IndexError, space.newtext("bytearray index out of range"))

  if nbytes < 8:
    intval = 0
    while end > begin:
      end -= 1
      intval = (intval << 8) + ord(ba_data[ end ])

    return W_SmallBits( nbytes<<3, intval )

  else:
    digits = []
    current_word = 0
    bitstart = 0

    while begin < end:
      item = ord(ba_data[begin])

      bitend = bitstart + 8

      if bitend <= SHIFT: # we are good
        current_word |= item << bitstart
        bitstart = bitend
      else:
        bitend -= SHIFT
        this_nbits = 8 - bitend
        current_word |= (item & get_int_mask(this_nbits) ) << bitstart
        digits.append(_store_digit(current_word))

        current_word = item >> this_nbits
        bitstart = 8 - this_nbits

      begin += 1

    digits.append(_store_digit(current_word))

    bigval = rbigint(digits[:], sign=1) # 1 is positive!!!
    bigval._normalize()

    return W_BigBits( nbytes<<3, bigval )

def read_bytearray_bits(space, w_arr, w_addr, w_nbytes):
  """read_bytearray_bits( bytearray, addr, nbytes )"""
  return read_bytearray_bits_impl( space, w_arr, w_addr, w_nbytes )

# write_bytearray_bits

def write_bytearray_bits_impl( space, w_arr, w_addr, w_nbytes, w_data ):
  # We directly manipulate bytearray here
  assert isinstance( w_arr, W_BytearrayObject )
  ba_data   = w_arr._data
  ba_len    = len(ba_data)
  ba_offset = w_arr._offset

  nbytes = 0
  if   isinstance(w_nbytes, W_SmallBits):
    nbytes = w_nbytes.intval
  elif type(w_nbytes) is W_IntObject:
    nbytes = w_nbytes.intval
  elif isinstance(w_nbytes, W_BigBits):
    tmp = w_nbytes.bigval
    if tmp.numdigits() > 1:
      raise oefmt(space.w_ValueError, "nbytes [%s] too big for bytearray write Bits%d",
                                      rbigint.str(tmp), w_nbytes.nbits )
    nbytes = tmp.digit(0)
  elif type(w_nbytes) is W_LongObject:
    nbytes = w_nbytes.num.toint()
  else:
    raise oefmt(space.w_TypeError, "Please pass in int/Bits" )

  addr = 0
  if   isinstance(w_addr, W_SmallBits):
    addr = w_addr.intval
  elif type(w_addr) is W_IntObject:
    addr = w_addr.intval
  elif isinstance(w_addr, W_BigBits):
    tmp = w_addr.bigval
    if tmp.numdigits() > 1:
      raise oefmt(space.w_ValueError, "Index [%s] too big for bytearray write Bits%d",
                                      rbigint.str(tmp), w_addr.nbits )
    addr = tmp.digit(0)
  elif type(w_addr) is W_LongObject:
    addr = w_addr.num.toint()
  else:
    raise oefmt(space.w_TypeError, "Please pass in int/Bits" )

  if addr < 0:
    raise oefmt(space.w_ValueError, "write_bytearray_bits only accept positive addr.")

  begin = addr + ba_offset
  end   = begin + nbytes

  if end > ba_len:
    raise OperationError(space.w_IndexError, space.newtext("bytearray index out of range"))

  if isinstance(w_data, W_SmallBits):
    if w_data.nbits > (nbytes << 3):
      raise oefmt(space.w_ValueError, "The value to write is Bits%d which is wider than the requested %d bytes.",
                                      w_data.nbits, nbytes )

    intval = w_data.intval
    while begin < end:
      ba_data[begin] = chr(intval & 255)
      intval = intval >> 8
      begin += 1

  elif isinstance(w_data, W_BigBits):
    if w_data.nbits > (nbytes << 3):
      raise oefmt(space.w_ValueError, "The value to write is Bits%d which is wider than the requested %d bytes.",
                                      w_data.nbits, nbytes )

    bigval = w_data.bigval

    # zero -- clear
    if not bigval.sign:
      while begin < end:
        ba_data[begin] = chr(0)
        begin += 1
      return

    # not zero
    size   = bigval.numdigits()
    curval = bigval.digit(0)
    curbit = curdigit = 0

    while begin < end:
      bitend = curbit + 8

      if bitend <= SHIFT:
        ba_data[begin] = chr(curval & 255)
        curval >>= 8
        curbit = bitend
        begin += 1

      else:
        curdigit += 1
        if curdigit >= size:
          ba_data[begin] = chr(curval)
          begin += 1
          while begin < end:
            ba_data[begin] = chr(0)
            begin += 1
          return

        else:
          nextval = bigval.digit(curdigit)
          bitend -= SHIFT
          this_nbits = 8 - bitend
          curval |= (nextval & get_int_mask(bitend)) << this_nbits
          ba_data[begin] = chr(curval)

          curval = nextval >> bitend
          curbit = bitend
          begin += 1

  else:
    raise oefmt(space.w_ValueError, "write_bytearray_bits only accepts Bits as data")

def write_bytearray_bits(space, w_arr, w_addr, w_nbytes, w_data):
  """write_bytearray_bits( bytearray, addr, nbytes )"""
  return write_bytearray_bits_impl( space, w_arr, w_addr, w_nbytes, w_data )
