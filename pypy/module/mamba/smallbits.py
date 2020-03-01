
import operator

from rpython.rlib import jit
from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rbigint     import rbigint, NULLDIGIT, SHIFT, BASE16
from rpython.tool.sourcetools import func_renamer, func_with_new_name

from pypy.interpreter.baseobjspace import W_Root
from pypy.interpreter.gateway import WrappedDefault, interp2app, interpindirect2app, unwrap_spec
from pypy.interpreter.error import OperationError, oefmt
from pypy.interpreter.typedef import TypeDef, GetSetProperty
from pypy.objspace.std.intobject import W_IntObject, wrapint, ovfcheck, _hash_int, SENTINEL
from pypy.objspace.std.longobject import W_LongObject, newlong, _hash_long
from pypy.objspace.std.sliceobject import W_SliceObject
from pypy.objspace.std.util import BINARY_OPS, CMP_OPS, COMMUTATIVE_OPS

@jit.elidable
def int_bit_length( val ):
  bits = 0
  if val < 0:
    val = -((val + 1) >> 1)
    bits = 1
  while val:
    bits += 1
    val >>= 1
  return bits

# Shunning:
# I don't wrap int/long anymore, just keep rbigint all the time
# Currently we only support arithmetic operations between Bits only, so
# really those radd/rsub methods don't take effect.

# NOTE that we should keep self.value positive after any computation:
# - The sign of the rbigint field should always be one
# - Always AND integer value with mask, never store any negative int
# * Performing rbigint.and_/rbigint.int_and_ will turn sign back to 1
# - rbigint._normalize() can only be called in @jit.elidable funcs

mask = rbigint([NULLDIGIT], 1, 1)
LONG_MASKS = [ mask ]
for i in xrange(1024):
  mask = mask.int_mul(2).int_add( 1 )
  LONG_MASKS.append( mask )

def get_long_mask( i ):
  return LONG_MASKS[ i ]
get_long_mask._always_inline_ = True

def get_int_mask( i ):
  return int((1<<i) - 1)
get_int_mask._always_inline_ = True

# This function implements fast ANDing mask functionality. PLEASE ONLY USE
# IT ON POSITIVE RBIGINT. On the other hand, AND get_long_mask for those
# operators that might produce negative results such as sub/new

@jit.elidable
def _rbigint_maskoff_high( value, masklen ):
  if not value.sign:  return value
  # assert value.sign > 0

  lastword = (masklen - 1)/SHIFT
  masksize  = lastword + 1
  if masksize > value.numdigits(): return value
  assert masksize > 0 # tell the dumb translator
  # From now on 0 < masksize <= value.numdigits(), so lastword exists

  ret = rbigint(value._digits[:masksize], 1, masksize)

  # Here, if masklen % SHIFT == 0, then we don't need to mask the last
  # word because wordpos = masklen / SHIFT = masksize in this case
  maskbit = masklen % SHIFT
  if maskbit != 0:
    lastdigit = ret.digit(lastword)
    mask = get_int_mask(maskbit)
    if lastdigit >= mask:
      ret.setdigit( lastword, lastdigit & mask )

  ret._normalize()
  return ret

#-------------------------------------------------------------------------
# Shunning: The following functions are specialized implementations for
# Bits arithmetics. Basically we squash arithmetic ops and ANDing mask to
# the same function to avoid copying and also reduce the constant factor
# based on the return type (int/rbigint).
#-------------------------------------------------------------------------

cmp_opp = {
  'lt': 'gt',
  'le': 'ge',
  'eq': 'eq', # commutative
  'ne': 'ne', # commutative
  'gt': 'lt',
  'ge': 'le',
}

class W_AbstractBits(W_Root):
  __slots__ = 'nbits'
  _immutable_fields_ = ['nbits']

  # mimic AbstractInt

  def _self_unaryop(opname, doc=None):
    @func_renamer('descr_' + opname)
    def descr_unaryop(self, space):
      raise NotImplementedError
    descr_unaryop.__doc__ = doc
    return descr_unaryop

  descr_pos = _self_unaryop('pos', "x.__pos__() <==> +x")
  descr_index = _self_unaryop('index',
                              "x[y:z] <==> x[y.__index__():z.__index__()]")
  descr_trunc = _self_unaryop('trunc',
                              "Truncating an Integral returns itself.")
  descr_floor = _self_unaryop('floor', "Flooring an Integral returns itself.")
  descr_ceil = _self_unaryop('ceil', "Ceiling of an Integral returns itself.")

  def _abstract_unaryop(opname, doc=SENTINEL):
    if doc is SENTINEL:
      doc = 'x.__%s__() <==> %s(x)' % (opname, opname)
    @func_renamer('descr_' + opname)
    def descr_unaryop(self, space):
      raise NotImplementedError
    descr_unaryop.__doc__ = doc
    return descr_unaryop

  descr_repr = _abstract_unaryop('repr')
  descr_str = _abstract_unaryop('str')

  descr_hash = _abstract_unaryop('hash')
  descr_getnewargs = _abstract_unaryop('getnewargs', None)
  descr_float = _abstract_unaryop('float')
  descr_neg = _abstract_unaryop('neg', "x.__neg__() <==> -x")
  descr_abs = _abstract_unaryop('abs')
  descr_bool = _abstract_unaryop('bool', "x.__bool__() <==> x != 0")
  descr_invert = _abstract_unaryop('invert', "x.__invert__() <==> ~x")

  def _abstract_cmpop(opname):
    @func_renamer('descr_' + opname)
    def descr_cmp(self, space, w_other):
      raise NotImplementedError
    descr_cmp.__doc__ = 'x.__%s__(y) <==> x%sy' % (opname, CMP_OPS[opname])
    return descr_cmp

  descr_lt = _abstract_cmpop('lt')
  descr_le = _abstract_cmpop('le')
  descr_eq = _abstract_cmpop('eq')
  descr_ne = _abstract_cmpop('ne')
  descr_gt = _abstract_cmpop('gt')
  descr_ge = _abstract_cmpop('ge')

  def _abstract_binop(opname):
    oper = BINARY_OPS.get(opname)
    if oper == '%':
      oper = '%%'
    oper = '%s(%%s, %%s)' % opname if not oper else '%%s%s%%s' % oper
    @func_renamer('descr_' + opname)
    def descr_binop(self, space, w_other):
      raise NotImplementedError

    descr_binop.__doc__ = "x.__%s__(y) <==> %s" % (opname,
                                                   oper % ('x', 'y'))
    descr_rbinop = func_with_new_name(descr_binop, 'descr_r' + opname)
    descr_rbinop.__doc__ = "x.__r%s__(y) <==> %s" % (opname,
                                                     oper % ('y', 'x'))
    return descr_binop, descr_rbinop

  descr_add, descr_radd = _abstract_binop('add')
  descr_sub, descr_rsub = _abstract_binop('sub')
  descr_mul, descr_rmul = _abstract_binop('mul')
  descr_matmul, descr_rmatmul = _abstract_binop('matmul')

  descr_and, descr_rand = _abstract_binop('and')
  descr_or, descr_ror = _abstract_binop('or')
  descr_xor, descr_rxor = _abstract_binop('xor')

  descr_lshift, descr_rlshift = _abstract_binop('lshift')
  descr_rshift, descr_rrshift = _abstract_binop('rshift')

  descr_floordiv, descr_rfloordiv = _abstract_binop('floordiv')
  descr_truediv, descr_rtruediv = _abstract_binop('truediv')
  descr_mod, descr_rmod = _abstract_binop('mod')
  descr_divmod, descr_rdivmod = _abstract_binop('divmod')

  def descr_pow(self, space, w_exponent, w_modulus=None):
    """x.__pow__(y[, z]) <==> pow(x, y[, z])"""
    raise NotImplementedError
  descr_rpow = func_with_new_name(descr_pow, 'descr_rpow')
  descr_rpow.__doc__ = "y.__rpow__(x[, z]) <==> pow(x, y[, z])"

  # value can be negative! Be extremely cautious with _rb_maskoff_high
  @staticmethod
  @unwrap_spec(w_value=WrappedDefault(0))
  def descr_new( space, w_objtype, w_nbits, w_value ):
    from pypy.module.mamba.bigbits import W_BigBits

    if type(w_nbits) is W_IntObject:
      nbits = w_nbits.intval

      if nbits <= SHIFT:
        if nbits < 1:
          raise oefmt(space.w_ValueError, "1 <= 'nbits' <= 512, not %d", w_nbits.intval)

        ret = space.allocate_instance( W_SmallBits, w_objtype )
        ret.nbits = nbits

        mask = get_int_mask(nbits)
        if   isinstance(w_value, W_IntObject):  ret.intval = w_value.intval & mask
        elif isinstance(w_value, W_SmallBits):  ret.intval = w_value.intval & mask
        elif isinstance(w_value, W_BigBits):    ret.intval = w_value.bigval.digit(0) & mask
        elif type(w_value) is W_LongObject:     ret.intval = w_value.num.int_and_( mask ).digit(0)
        else:
          raise oefmt(space.w_TypeError, "Value used to construct Bits%d "
                      "must be int/long/Bits " # or whatever has __int__, "
                      "not '%T'", nbits, w_value)

      else: # nbits > SHIFT
        if nbits > 512:
          raise oefmt(space.w_ValueError, "1 <= 'nbits' <= 512, not %d", w_nbits.intval)

        ret = space.allocate_instance( W_BigBits, w_objtype )
        ret.nbits = nbits

        if   isinstance(w_value, W_IntObject):  ret.bigval = get_long_mask(nbits).int_and_( w_value.intval )
        elif isinstance(w_value, W_SmallBits):  ret.bigval = rbigint.fromint( w_value.intval )
        elif isinstance(w_value, W_BigBits):    ret.bigval = _rbigint_maskoff_high( w_value.bigval, nbits )
        elif type(w_value) is W_LongObject:     ret.bigval = get_long_mask(nbits).and_( w_value.num )
        else:
          raise oefmt(space.w_TypeError, "Value used to construct Bits%d "
                      "must be int/long/Bits" # or whatever has __int__, "
                      "not '%T'", nbits, w_value)

    else:
      raise oefmt(space.w_TypeError, "'nbits' must be an int, not '%T'", w_nbits )
    return ret

  def check_slice_range( self, space, start, stop ):
    if start >= stop:
      raise oefmt(space.w_ValueError, "Invalid range: start [%d] >= stop [%d]", start, stop )
    if start < 0:
      raise oefmt(space.w_ValueError, "Negative start: [%d]", start )
    if stop > self.nbits:
      raise oefmt(space.w_ValueError, "Stop [%d] too big for Bits%d", stop, self.nbits )

  check_slice_range._always_inline_ = True

  # Bits specific

  def descr_get_nbits(self, space):
    return wrapint( space, self.nbits )

  def descr_uint(self, space):
    raise NotImplementedError

  def descr_int(self, space):
    raise NotImplementedError

  def descr_flip(self, space):
    raise NotImplementedError

  def descr_ilshift(self, space, w_other):
    return self._descr_ilshift(space, w_other)

  def descr_copy( self, space ):
    raise NotImplementedError

  def descr_deepcopy( self, space ):
    raise NotImplementedError

  def descr_clone( self, space ):
    raise NotImplementedError

  def descr_getitem(self, space, w_index):
    raise NotImplementedError

  def descr_setitem(self, space, w_index, w_value):
    raise NotImplementedError

class W_SmallBits(W_AbstractBits):
  __slots__ = ( "nbits", "intval" )
  _immutable_fields_ = [ "nbits" ]

  def __init__( self, nbits, intval=0 ):
    self.nbits  = nbits
    self.intval = intval

  def descr_copy(self):
    return W_SmallBits( self.nbits, self.intval )

  descr_deepcopy = func_with_new_name(descr_copy, 'descr_deepcopy')
  descr_clone    = func_with_new_name(descr_copy, 'descr_clone')

  #-----------------------------------------------------------------------
  # get/setitem
  #-----------------------------------------------------------------------

  def descr_getitem(self, space, w_index):
    from pypy.module.mamba.bigbits import W_BigBits

    if type(w_index) is W_SliceObject:
      if space.is_w(w_index.w_step, space.w_None):
        w_start = w_index.w_start
        start   = 0

        if type(w_start) is W_IntObject:
          start = w_start.intval
        elif isinstance(w_start, W_SmallBits):
          start = w_start.intval
        elif isinstance(w_start, W_BigBits):
          tmp = w_start.bigval
          if tmp.numdigits() > 1:
            raise oefmt(space.w_ValueError, "Index [%s] too big for Bits%d",
                                            rbigint.str(tmp), self.nbits )
          start = tmp.digit(0)
        elif type(w_start) is W_LongObject:
          start = w_start.num.toint()
        else:
          raise oefmt(space.w_TypeError, "Please pass in int/Bits variables for the slice. -- getitem #1" )

        w_stop = w_index.w_stop
        stop   = 0
        if type(w_stop) is W_IntObject:
          stop = w_stop.intval
        elif isinstance(w_stop, W_SmallBits):
          stop = w_stop.intval
        elif isinstance(w_stop, W_BigBits):
          tmp = w_stop.bigval
          if tmp.numdigits() > 1:
            raise oefmt(space.w_ValueError, "Index [%s] too big for Bits%d",
                                            rbigint.str(tmp), self.nbits )
          stop = tmp.digit(0)
        elif type(w_stop) is W_LongObject:
          stop = w_stop.num.toint()
        else:
          raise oefmt(space.w_TypeError, "Please pass in int/Bits variables for the slice. -- getitem #2" )

        self.check_slice_range( space, start, stop )
        slice_nbits = stop - start

        res = (self.intval >> start) & get_int_mask(slice_nbits)
        return W_SmallBits( slice_nbits, res )

      else:
        raise oefmt(space.w_ValueError, "Bits slice cannot have step." )

    else:
      index = 0
      if   type(w_index) is W_IntObject:
        index = w_index.intval
      elif isinstance(w_index, W_SmallBits):
        index = w_index.intval
        if index < 0:
          raise oefmt(space.w_ValueError, "Negative index: [%d]", index )
      elif isinstance(w_index, W_BigBits):
        tmp = w_index.bigval
        if tmp.numdigits() > 1:
          raise oefmt(space.w_ValueError, "Index [%s] too big for Bits%d",
                                          rbigint.str(tmp), self.nbits )
        index = tmp.digit(0)
      elif type(w_index) is W_LongObject:
        index = w_index.num.toint()
        if index < 0:
          raise oefmt(space.w_ValueError, "Negative index: [%d]", index )
      else:
        raise oefmt(space.w_TypeError, "Please pass in int/Bits variables for the slice. -- getitem #3" )

      if index >= self.nbits:
        raise oefmt(space.w_ValueError, "Index [%d] too big for Bits%d", index, self.nbits )

      return W_SmallBits( 1, (self.intval >> index) & 1 )

  def descr_setitem(self, space, w_index, w_other):
    from pypy.module.mamba.bigbits import W_BigBits

    if type(w_index) is W_SliceObject:
      if space.is_w(w_index.w_step, space.w_None):
        w_start = w_index.w_start
        start   = 0

        if type(w_start) is W_IntObject:
          start = w_start.intval
        elif isinstance(w_start, W_SmallBits):
          start = w_start.intval
        elif isinstance(w_start, W_BigBits):
          tmp = w_start.bigval
          if tmp.numdigits() > 1:
            raise oefmt(space.w_ValueError, "Index [%s] too big for Bits%d",
                                            rbigint.str(tmp), self.nbits )
          start = tmp.digit(0)
        elif type(w_start) is W_LongObject:
          start = w_start.num.toint()
        else:
          raise oefmt(space.w_TypeError, "Please pass in int/Bits variables for the slice. -- setitem #1" )

        w_stop = w_index.w_stop
        stop   = 0
        if type(w_stop) is W_IntObject:
          stop = w_stop.intval
        elif isinstance(w_stop, W_SmallBits):
          stop = w_stop.intval
        elif isinstance(w_stop, W_BigBits):
          tmp = w_stop.bigval
          if tmp.numdigits() > 1:
            raise oefmt(space.w_ValueError, "Index [%s] too big for Bits%d",
                                            rbigint.str(tmp), self.nbits )
          stop = tmp.digit(0)
        elif type(w_stop) is W_LongObject:
          stop = w_stop.num.toint()
        else:
          raise oefmt(space.w_TypeError, "Please pass in int/Bits variables for the slice. -- setitem #2" )

        self.check_slice_range( space, start, stop )
        slice_nbits = stop - start

        # Check value bitlen. No need to check Bits, but check int/long.

        if isinstance(w_other, W_SmallBits):
          if w_other.nbits > slice_nbits:
            raise oefmt(space.w_ValueError, "Value of type Bits%d cannot fit into "
                        "[%d:%d](%d-bit) slice", w_other.nbits, start, stop, slice_nbits )
          valuemask   = ~(get_int_mask(slice_nbits) << start)
          self.intval = (self.intval & valuemask) | (w_other.intval << start)

        elif isinstance(w_other, W_BigBits):
          raise oefmt(space.w_ValueError, "Value of type Bits%d cannot fit into "
                      "[%d:%d](%d-bit) slice", w_other.nbits, start, stop, slice_nbits )

        elif isinstance(w_other, W_IntObject):
          other = w_other.intval
          blen  = int_bit_length( other )
          if blen > slice_nbits:
            raise oefmt(space.w_ValueError, "Value %d cannot fit into "
                  "[%d:%d] (%d-bit) slice", other, start, stop, slice_nbits )

          mask = get_int_mask(slice_nbits)
          other &= mask
          valuemask = ~(mask << start)
          self.intval = (self.intval & valuemask) | (other << start)

        elif type(w_other) is W_LongObject:
          other = w_other.num
          blen = other.bit_length()
          if blen > slice_nbits:
            raise oefmt(space.w_ValueError, "Value %s cannot fit into "
                  "[%d:%d] (%d-bit) slice", rbigint.str(other), start, stop, slice_nbits )
          mask = get_int_mask(slice_nbits)
          other = other.int_and_(mask).digit(0)
          valuemask = ~(mask << start)
          self.intval = (self.intval & valuemask) | (other << start)

      else:
        raise oefmt(space.w_ValueError, "Bits slice cannot have step." )

    else:
      index = 0
      if   type(w_index) is W_IntObject:
        index = w_index.intval
      elif isinstance(w_index, W_SmallBits):
        index = w_index.intval
        if index < 0:
          raise oefmt(space.w_ValueError, "Negative index: [%d]", index )
      elif isinstance(w_index, W_BigBits):
        tmp = w_index.bigval
        if tmp.numdigits() > 1:
          raise oefmt(space.w_ValueError, "Index [%s] too big for Bits%d",
                                          rbigint.str(tmp), self.nbits )
        index = tmp.digit(0)
      elif type(w_index) is W_LongObject:
        index = w_index.num.toint()
        if index < 0:
          raise oefmt(space.w_ValueError, "Negative index: [%d]", index )
      else:
        raise oefmt(space.w_TypeError, "Please pass in int/Bits variables for the slice. -- setitem #3" )

      if index >= self.nbits:
        raise oefmt(space.w_ValueError, "Index [%d] too big for Bits%d", index, self.nbits )

      # Check value bitlen. No need to check Bits, but check int/long.
      if isinstance(w_other, W_SmallBits):
        o_nbits = w_other.nbits
        if o_nbits > 1:
          raise oefmt(space.w_ValueError, "Bits%d cannot fit into 1-bit slice", o_nbits )
        other = w_other.intval # must be 1-bit and don't even check

        self.intval = (self.intval & ~(1 << index)) | (other << index)

      elif isinstance(w_other, W_IntObject):
        other = w_other.intval
        if other < 0 or other > 1:
          raise oefmt(space.w_ValueError, "Value %d cannot fit into 1-bit slice", other )

        self.intval = (self.intval & ~(1 << index)) | (other << index)

      # Check value bitlen. No need to check Bits, but check int/long.
      elif isinstance(w_other, W_BigBits):
        o_nbits = w_other.nbits
        if o_nbits > 1:
          raise oefmt(space.w_ValueError, "Bits%d cannot fit into 1-bit slice", o_nbits )
        other = w_other.intval # must be 1-bit and don't even check

      elif type(w_other) is W_LongObject:
        other = w_other.num
        if other.numdigits() > 1:
          raise oefmt(space.w_ValueError, "Value %s cannot fit into 1-bit slice", rbigint.str(other) )

        other = other.digit(0)
        if other < 0 or other > 1:
          raise oefmt(space.w_ValueError, "Value %d cannot fit into 1-bit slice", other )

        self.intval = (self.intval & ~(1 << index)) | (other << index)
      else:
        raise oefmt(space.w_TypeError, "Please pass in int/long/Bits value. -- setitem #4" )

  #-----------------------------------------------------------------------
  # Miscellaneous methods for string format
  #-----------------------------------------------------------------------

  def _format16(self, space):
    data = (rbigint.fromint(self.intval)).format(BASE16)
    w_data = space.newtext( data )
    return space.text_w( w_data.descr_zfill(space, (((self.nbits-1)>>2)+1)) )

  def descr_repr(self, space):
    return space.newtext( "Bits%d( 0x%s )" % (self.nbits, self._format16(space)) )

  def descr_str(self, space):
    return space.newtext( "%s" % (self._format16(space)) )

  #-----------------------------------------------------------------------
  # comparators
  #-----------------------------------------------------------------------

  def _make_descr_cmp(opname):
    iiop = getattr( operator, opname )
    llop = getattr( rbigint , opname )
    liop = getattr( rbigint , "int_"+opname )
    ilopp = getattr( rbigint , "int_"+cmp_opp[opname] )

    @func_renamer('descr_' + opname)
    def descr_cmp(self, space, w_other):
      from pypy.module.mamba.bigbits import W_BigBits
      x = self.intval
      if   isinstance(w_other, W_SmallBits):
        return W_SmallBits( 1, iiop( x, w_other.intval ) )

      elif isinstance(w_other, W_IntObject):
        # TODO Maybe add int_bit_length check?
        return W_SmallBits( 1, iiop( x, w_other.intval & get_int_mask(self.nbits) ) )

      elif isinstance(w_other, W_BigBits):
        return W_SmallBits( 1, ilopp( w_other.bigval, x ) )

      elif type(w_other) is W_LongObject:
        # TODO Maybe add bit_length check?
        return W_SmallBits( 1, ilopp( get_long_mask(self.nbits).and_( w_other.num ), x ) )
      # Match cpython behavior
      return W_SmallBits( 1, 0 )
      # raise oefmt(space.w_TypeError, "Please compare two Bits/int/long objects" )

    return descr_cmp

  descr_lt = _make_descr_cmp('lt')
  descr_le = _make_descr_cmp('le')
  descr_eq = _make_descr_cmp('eq')
  descr_ne = _make_descr_cmp('ne')
  descr_gt = _make_descr_cmp('gt')
  descr_ge = _make_descr_cmp('ge')

  #-----------------------------------------------------------------------
  # binary arith ops
  #-----------------------------------------------------------------------
  # Note that we have to check commutativity along with type because
  # rbigint doesn't have "rsub" implementation so we cannot do "int"-"long"

  def _make_descr_binop_opname(opname, ovf=True):
    # Shunning: shouldn't overwrite opname -- "and_" is not in COMMUTATIVE_OPS
    _opn = opname + ('_' if opname in ('and', 'or') else '')
    llop = getattr( rbigint, _opn )
    liop = getattr( rbigint, "int_"+_opn )
    iiop = getattr( operator, _opn )

    @func_renamer('descr_' + opname)
    def descr_binop(self, space, w_other):
      from pypy.module.mamba.bigbits import W_BigBits
      # add, sub, mul
      if ovf:
        x = self.intval

        if isinstance(w_other, W_SmallBits):
          y = w_other.intval
          res_nbits = max(self.nbits, w_other.nbits)
          mask = get_int_mask(res_nbits)
          try:
            z = ovfcheck( iiop(x, y) )
            return W_SmallBits( res_nbits, z & mask )
          except OverflowError:
            z = liop( rbigint.fromint(x), y )
            if opname in COMMUTATIVE_OPS: # add, mul
              z = z.digit(0) & mask
            else: # sub, should AND mask
              z = z.int_and_( mask ).digit(0)
            return W_SmallBits( res_nbits, z )

        elif isinstance(w_other, W_IntObject):
          y = w_other.intval
          mask = get_int_mask(self.nbits)
          try:
            z = ovfcheck( iiop(x, y) )
            return W_SmallBits( self.nbits, z & mask )
          except OverflowError:
            z = liop( rbigint.fromint(x), y )
            if opname in COMMUTATIVE_OPS: # add, mul
              z = z.digit(0) & mask
            else: # sub, should AND mask
              z = z.int_and_( mask ).digit(0)
            return W_SmallBits( self.nbits, z )

        elif isinstance(w_other, W_BigBits):
          y = w_other.bigval
          if opname in COMMUTATIVE_OPS: # add, mul
            z = _rbigint_maskoff_high( liop(y, x), w_other.nbits )
            return W_BigBits( w_other.nbits, z )
          else: # sub, should AND get_long_mask
            z = llop( rbigint.fromint(x), y )
            z = z.and_( get_long_mask(w_other.nbits) )
            return W_BigBits( w_other.nbits, z )

        elif type(w_other) is W_LongObject:
          y = w_other.num
          mask = get_int_mask(self.nbits)
          if opname in COMMUTATIVE_OPS: # add, mul
            z = liop(y, x).int_and_( mask )
            return W_SmallBits( self.nbits, z.digit(0) )
          else: # sub
            z = llop( rbigint.fromint(x), y ).int_and_( mask )
            return W_SmallBits( self.nbits, z.digit(0) )

      # and, or, xor, no overflow
      # opname should be in COMMUTATIVE_OPS
      else:
        x = self.intval
        if isinstance(w_other, W_SmallBits):
          return W_SmallBits( max(self.nbits, w_other.nbits), iiop( x, w_other.intval ) )

        elif isinstance(w_other, W_IntObject): # TODO Maybe add int_bit_length check?
          return W_SmallBits( self.nbits, iiop( x, w_other.intval ) )

        elif isinstance(w_other, W_BigBits):
          return W_BigBits( w_other.nbits, liop( w_other.bigval, x ) )

        elif type(w_other) is W_LongObject: # TODO Maybe add int_bit_length check?
          return W_BigBits( self.nbits, liop( w_other.num, x ) )

      raise oefmt(space.w_TypeError, "Please do %s between Bits and Bits/int/long objects", opname)

    if opname in COMMUTATIVE_OPS:
      @func_renamer('descr_r' + opname)
      def descr_rbinop(self, space, w_other):
        return descr_binop(self, space, w_other)
      return descr_binop, descr_rbinop

    # TODO sub
    @func_renamer('descr_r' + opname)
    def descr_rbinop(self, space, w_other):
      raise oefmt(space.w_TypeError, "r%s not implemented", opname )
    return descr_binop, descr_rbinop

  # Special rsub ..
  def descr_rsub( self, space, w_other ):
    llop = getattr( rbigint, "sub" )
    liop = getattr( rbigint, "int_sub" )
    iiop = getattr( operator, "sub" )

    y = self.intval

    if isinstance(w_other, W_IntObject):
      x = w_other.intval
      mask = get_int_mask(self.nbits)
      try:
        z = ovfcheck( iiop(x, y) )
        return W_SmallBits( self.nbits, z & mask )
      except OverflowError:
        z = liop( rbigint.fromint(x), y ).int_and_( mask ).digit(0)
        return W_SmallBits( self.nbits, z )
    elif type(w_other) is W_LongObject:
      x = w_other.num
      z = liop( x, y ).int_and_( get_int_mask(self.nbits) )
      return W_SmallBits( self.nbits, z.digit(0) )

  descr_add, descr_radd = _make_descr_binop_opname('add')
  descr_sub, _          = _make_descr_binop_opname('sub')
  descr_mul, descr_rmul = _make_descr_binop_opname('mul')

  descr_and, descr_rand = _make_descr_binop_opname('and', ovf=False)
  descr_or, descr_ror   = _make_descr_binop_opname('or', ovf=False)
  descr_xor, descr_rxor = _make_descr_binop_opname('xor', ovf=False)

  def descr_rshift(self, space, w_other):
    from pypy.module.mamba.bigbits import W_BigBits

    x = self.intval
    if isinstance(w_other, W_SmallBits):
      shamt = w_other.intval
      if shamt <= SHIFT:  return W_SmallBits( self.nbits, x >> shamt )
      return W_SmallBits( self.nbits )

    elif isinstance(w_other, W_IntObject):
      shamt = w_other.intval
      if shamt < 0: raise oefmt( space.w_ValueError, "negative shift amount" )
      if shamt <= SHIFT:  return W_SmallBits( self.nbits, x >> shamt )
      return W_SmallBits( self.nbits )

    elif isinstance(w_other, W_BigBits):
      big = w_other.bigval
      shamt = big.digit(0)
      if big.numdigits() == 1 and shamt <= SHIFT:
        return W_SmallBits( self.nbits, x >> shamt )
      return W_SmallBits( self.nbits )

    elif type(w_other) is W_LongObject:
      big = w_other.num
      if big.sign < 0: raise oefmt( space.w_ValueError, "negative shift amount" )
      shamt = big.digit(0)
      if big.numdigits() == 1 and shamt <= SHIFT:
        return W_SmallBits( self.nbits, x >> shamt )
      return W_SmallBits( self.nbits )

    raise oefmt(space.w_TypeError, "Please do rshift between <Bits, Bits/int/long> objects" )

  def descr_rrshift(self, space, w_other): # int >> bits, what is nbits??
    raise oefmt(space.w_TypeError, "rrshift not implemented" )

  def descr_lshift(self, space, w_other):
    from pypy.module.mamba.bigbits import W_BigBits

    x = self.intval

    if isinstance(w_other, W_SmallBits):
      shamt = w_other.intval
      if shamt >= self.nbits:  return W_SmallBits( self.nbits, 0 )
      return W_SmallBits( self.nbits, (x & get_int_mask(self.nbits - shamt)) << shamt )

    elif isinstance(w_other, W_IntObject):
      shamt = w_other.intval
      if shamt < 0: raise oefmt( space.w_ValueError, "negative shift amount" )
      if shamt >= self.nbits:  return W_SmallBits( self.nbits )
      return W_SmallBits( self.nbits, (x & get_int_mask(self.nbits - shamt)) << shamt )

    elif isinstance(w_other, W_BigBits):
      big = w_other.bigval
      shamt = big.digit(0)
      if big.numdigits() == 1 and shamt <= self.nbits:
        return W_SmallBits( self.nbits, (x & get_int_mask(self.nbits - shamt)) << shamt )
      return W_SmallBits( self.nbits )

    elif type(w_other) is W_LongObject:
      big = w_other.num
      if big.sign < 0: raise oefmt( space.w_ValueError, "negative shift amount" )
      shamt = big.digit(0)
      if big.numdigits() == 1 and shamt <= self.nbits:
        return W_SmallBits( self.nbits, (x & get_int_mask(self.nbits - shamt)) << shamt )
      return W_SmallBits( self.nbits )

    raise oefmt(space.w_TypeError, "Please do lshift between <Bits, Bits/int/long> objects" )

  def descr_rlshift(self, space, w_other): # int << Bits, what is nbits??
    raise oefmt(space.w_TypeError, "rlshift not implemented" )

  #-----------------------------------------------------------------------
  # <<=
  #-----------------------------------------------------------------------

  def _descr_ilshift(self, space, w_other):
    from pypy.module.mamba.bigbits import W_BigBits

    if isinstance(w_other, W_SmallBits):
      if self.nbits != w_other.nbits:
        raise oefmt(space.w_ValueError, "Bitwidth mismatch during <<=, assigning Bits%d <<= Bits%d",
                                        self.nbits, w_other.nbits)
      return W_SmallBitsWithNext( self.nbits, self.intval, w_other.intval )

    # elif type(w_other) is W_IntObject: # TODO check int_bit_length?
      # return W_SmallBitsWithNext( self.nbits, self.intval, w_other.intval & get_int_mask(self.nbits) )

    elif isinstance(w_other, W_BigBits):
      raise oefmt(space.w_ValueError, "Bitwidth mismatch during <<=, assigning Bits%d <<= Bits%d",
                                      self.nbits, w_other.nbits)

    # elif type(w_other) is W_LongObject: # TODO check int_bit_length?
      # return W_SmallBitsWithNext( self.nbits, self.intval,
        # w_other.num.int_and_( get_int_mask(self.nbits) ).digit(0) )

    else:
      raise oefmt(space.w_TypeError, "RHS of <<= has to be Bits%d, not '%T'", self.nbits, w_other)

  def descr_ilshift(self, space, w_other):
    return self._descr_ilshift(space, w_other)

  def _descr_flip(self, space):
    raise oefmt(space.w_TypeError, "_flip cannot be called on '%T' objects which has no _next", self)

  def descr_flip(self, space):
    return self._descr_flip(space)

  #-----------------------------------------------------------------------
  # value access
  #-----------------------------------------------------------------------

  def descr_uint(self, space):
    return wrapint( space, self.intval )

  def descr_int(self, space): # TODO
    index = self.nbits - 1
    intval = self.intval
    msb = (intval >> index) & 1
    # if not msb: return wrapint( space, intval )
    # return wrapint( space, intval - get_int_mask(self.nbits) - 1 )
    return wrapint( space, intval - msb*get_int_mask(self.nbits) - msb )

  descr_pos = func_with_new_name( descr_uint, 'descr_pos' )
  descr_index = func_with_new_name( descr_uint, 'descr_index' )

  #-----------------------------------------------------------------------
  # unary ops
  #-----------------------------------------------------------------------

  def descr_bool(self, space):
    return space.newbool( self.intval != 0 )

  def descr_invert(self, space):
    return W_SmallBits( self.nbits, get_int_mask(self.nbits) - self.intval )

  # def descr_neg(self, space):

  def descr_hash(self, space):
    hash_nbits = _hash_int( self.nbits )
    hash_value = _hash_int( self.intval )

    # Manually implement a single iter of W_TupleObject.descr_hash

    x = 0x345678
    x = (x ^ hash_nbits) * 1000003
    x = (x ^ hash_value) * (1000003+82520+1+1)
    x += 97531
    return space.newint( intmask(x) )

#-----------------------------------------------------------------------
# Bits with next fields
#-----------------------------------------------------------------------

class W_SmallBitsWithNext(W_SmallBits):
  __slots__ = ( "nbits", "intval", "next_intval" )
  _immutable_fields_ = [ "nbits", "intval" ]

  def __init__( self, nbits, intval, next_intval):
    self.nbits  = nbits
    self.intval = intval
    self.next_intval = next_intval

  def descr_setitem(self, space, w_index, w_value):
    raise oefmt(space.w_TypeError, "You shouldn't do x[a:b]=y on flip-flop")

  def descr_copy( self ):
    return W_SmallBitsWithNext( self.nbits, self.intval, self.next_intval )

  def _descr_ilshift(self, space, w_other):
    from pypy.module.mamba.bigbits import W_BigBits

    if isinstance(w_other, W_SmallBits):
      if self.nbits != w_other.nbits:
        raise oefmt(space.w_ValueError, "Bitwidth mismatch during <<=, assigning Bits%d <<= Bits%d",
                                        self.nbits, w_other.nbits)
      self.next_intval = w_other.intval

    # elif type(w_other) is W_IntObject: # TODO check int_bit_length?
      # self.next_intval = w_other.intval & get_int_mask(self.nbits)

    elif isinstance(w_other, W_BigBits):
      raise oefmt(space.w_ValueError, "Bitwidth mismatch during <<=, assigning Bits%d <<= Bits%d",
                                      self.nbits, w_other.nbits)
    # elif type(w_other) is W_LongObject: # TODO check int_bit_length?
      # self.next_intval = w_other.num.int_and_( get_int_mask(self.nbits) ).digit(0)

    else:
      # raise oefmt(space.w_TypeError, "RHS of <<= has to be Bits or int, not '%T'", w_other)
      raise oefmt(space.w_TypeError, "RHS of <<= has to be Bits%d, not '%T'", self.nbits, w_other)

    return self

  def _descr_flip(self, space):
    self.intval = self.next_intval

W_AbstractBits.typedef = TypeDef("Bits",
    nbits = GetSetProperty(W_AbstractBits.descr_get_nbits),

    uint  = interp2app(W_AbstractBits.descr_uint),
    int   = interp2app(W_AbstractBits.descr_int),

    # Basic operations
    __new__ = interp2app(W_AbstractBits.descr_new),
    __getitem__ = interpindirect2app(W_AbstractBits.descr_getitem),
    __setitem__ = interpindirect2app(W_AbstractBits.descr_setitem),
    __copy__ = interpindirect2app(W_AbstractBits.descr_copy),
    __deepcopy__ = interpindirect2app(W_AbstractBits.descr_deepcopy),

    # String formats
    __hash__  = interpindirect2app(W_AbstractBits.descr_hash),
    __repr__ = interpindirect2app(W_AbstractBits.descr_repr),
    __str__  = interpindirect2app(W_AbstractBits.descr_str),
    __getnewargs__ = interpindirect2app(W_AbstractBits.descr_getnewargs),

    # Value access
    __int__   = interpindirect2app(W_AbstractBits.descr_uint), # TODO use uint now
    __index__ = interpindirect2app(W_AbstractBits.descr_index),
    __trunc__ = interpindirect2app(W_AbstractBits.descr_trunc),
    __float__ = interpindirect2app(W_AbstractBits.descr_float),
    # __round__ = interpindirect2app(W_AbstractBits.descr_round),

    # Unary ops
    __pos__   = interpindirect2app(W_AbstractBits.descr_pos),
    __neg__    = interpindirect2app(W_AbstractBits.descr_neg),
    __abs__    = interpindirect2app(W_AbstractBits.descr_abs),
    __bool__   = interpindirect2app(W_AbstractBits.descr_bool), # no __nonzero__ in Python3 anymore
    __invert__ = interpindirect2app(W_AbstractBits.descr_invert),
    __floor__ = interpindirect2app(W_AbstractBits.descr_floor),
    __ceil__ = interpindirect2app(W_AbstractBits.descr_ceil),

    # Comparators
    __lt__ = interpindirect2app(W_AbstractBits.descr_lt),
    __le__ = interpindirect2app(W_AbstractBits.descr_le),
    __eq__ = interpindirect2app(W_AbstractBits.descr_eq),
    __ne__ = interpindirect2app(W_AbstractBits.descr_ne),
    __gt__ = interpindirect2app(W_AbstractBits.descr_gt),
    __ge__ = interpindirect2app(W_AbstractBits.descr_ge),

    # Binary fast arith ops
    __add__  = interpindirect2app(W_AbstractBits.descr_add),
    __radd__ = interpindirect2app(W_AbstractBits.descr_radd),
    __sub__  = interpindirect2app(W_AbstractBits.descr_sub),
    __rsub__ = interpindirect2app(W_AbstractBits.descr_rsub),
    __mul__  = interpindirect2app(W_AbstractBits.descr_mul),
    __rmul__ = interpindirect2app(W_AbstractBits.descr_rmul),

    # Binary logic ops
    __and__  = interpindirect2app(W_AbstractBits.descr_and),
    __rand__ = interpindirect2app(W_AbstractBits.descr_rand),
    __or__   = interpindirect2app(W_AbstractBits.descr_or),
    __ror__  = interpindirect2app(W_AbstractBits.descr_ror),
    __xor__  = interpindirect2app(W_AbstractBits.descr_xor),
    __rxor__ = interpindirect2app(W_AbstractBits.descr_rxor),

    # Binary shift ops
    __lshift__  = interpindirect2app(W_AbstractBits.descr_lshift),
    __rlshift__ = interpindirect2app(W_AbstractBits.descr_rlshift),
    __rshift__  = interpindirect2app(W_AbstractBits.descr_rshift),
    __rrshift__ = interpindirect2app(W_AbstractBits.descr_rrshift),

    # Binary slow arith ops
    __floordiv__  = interpindirect2app(W_AbstractBits.descr_floordiv),
    __rfloordiv__ = interpindirect2app(W_AbstractBits.descr_rfloordiv),
    __truediv__   = interpindirect2app(W_AbstractBits.descr_truediv),
    __rtruediv__  = interpindirect2app(W_AbstractBits.descr_rtruediv),
    __mod__       = interpindirect2app(W_AbstractBits.descr_mod),
    __rmod__      = interpindirect2app(W_AbstractBits.descr_rmod),
    __divmod__    = interpindirect2app(W_AbstractBits.descr_divmod),
    __rdivmod__   = interpindirect2app(W_AbstractBits.descr_rdivmod),

    __pow__       = interpindirect2app(W_AbstractBits.descr_pow),
    __rpow__      = interpindirect2app(W_AbstractBits.descr_rpow),

    # PyMTL3 specific
    # <<=
    __ilshift__ = interpindirect2app(W_AbstractBits.descr_ilshift),
    _flip = interpindirect2app(W_AbstractBits.descr_flip),

    # clone
    clone = interpindirect2app(W_AbstractBits.descr_clone),
)
