import operator

from rpython.rlib import jit
from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rbigint     import rbigint, SHIFT, NULLDIGIT, ONERBIGINT, \
                                     NULLRBIGINT, _store_digit, _x_int_sub, \
                                     _widen_digit, BASE16
from rpython.tool.sourcetools import func_renamer, func_with_new_name

from pypy.module.mamba.smallbits import W_AbstractBits, W_SmallBits, int_bit_length, \
                                        get_long_mask, get_int_mask, _rbigint_maskoff_high, cmp_opp, \
                                        _get_index, _get_slice_range
from pypy.interpreter.baseobjspace import W_Root
from pypy.interpreter.gateway import WrappedDefault, interp2app, interpindirect2app, unwrap_spec
from pypy.interpreter.error import OperationError, oefmt
from pypy.objspace.std.intobject import W_IntObject, wrapint, ovfcheck, _hash_int
from pypy.objspace.std.longobject import W_LongObject, newlong, _hash_long
from pypy.objspace.std.sliceobject import W_SliceObject
from pypy.objspace.std.util import COMMUTATIVE_OPS

# This function implements rshift between two rbigints
@jit.elidable
def _rbigint_rshift( value, shamt ):
  if not value.sign or not shamt.sign:  return value
  if shamt.numdigits() > 1: return NULLRBIGINT
  shamt = shamt.digit(0)

  wordshift = shamt / SHIFT
  newsize = value.numdigits() - wordshift
  if newsize <= 0:  return NULLRBIGINT

  loshift = shamt - wordshift*SHIFT
  hishift = SHIFT - loshift
  ret = rbigint([NULLDIGIT]*newsize, 1, newsize)

  i = 0
  lastidx = newsize - 1
  curword = value.digit(wordshift)
  while i < lastidx:
    newdigit  = curword >> loshift
    wordshift = wordshift + 1
    curword   = value.digit(wordshift)
    ret.setdigit(i, newdigit | (curword << hishift) )
    i += 1
  # last digit
  ret.setdigit(i, curword >> loshift)

  ret._normalize()
  return ret
_rbigint_rshift._always_inline_ = 'try' # It's so fast that it's always benefitial.

# This function implements getslice functionality that returns rbigint.
@jit.elidable
def _rbigint_rshift_maskoff( value, shamt, masklen ):
  if not value.sign:  return value
  # shamt must be > 0, value.sign must > 0
  if shamt == 0: return _rbigint_maskoff_high( value, masklen )

  wordshift = shamt / SHIFT
  oldsize   = value.numdigits()
  if oldsize <= wordshift:  return NULLRBIGINT

  newsize  = oldsize - wordshift
  masksize = (masklen - 1)/SHIFT + 1
  retsize  = min( newsize, masksize )

  loshift = shamt - wordshift*SHIFT
  hishift = SHIFT - loshift
  ret = rbigint( [NULLDIGIT] * retsize, 1, retsize )
  i = 0
  while i < retsize:
    newdigit = (value.digit(wordshift) >> loshift)
    if i+1 < newsize:
      newdigit |= (value.digit(wordshift+1) << hishift)
    ret.setdigit(i, newdigit)
    i += 1
    wordshift += 1

  if masksize <= retsize:
    maskbit = masklen % SHIFT
    if maskbit != 0:
      lastword  = i - 1
      lastdigit = ret.digit(lastword)
      mask      = get_int_mask(maskbit)
      if lastdigit >= mask:
        ret.setdigit( lastword, lastdigit & mask )

  ret._normalize()
  return ret
_rbigint_rshift_maskoff._always_inline_ = True

# This function implements getslice functionality that returns normal int
# Shunning: This is FASTER than calling the above func and do x.digit(0)
@jit.elidable
def _rbigint_rshift_maskoff_retint( value, shamt, masklen ):
  # assert masklen <= SHIFT
  if not value.sign:  return 0
  # shamt must be > 0, value.sign must > 0
  if shamt == 0:
    return value.digit(0) & get_int_mask(masklen)

  wordshift = shamt / SHIFT
  oldsize   = value.numdigits()
  if oldsize <= wordshift:  return 0

  newsize = oldsize - wordshift
  loshift = shamt - wordshift*SHIFT
  hishift = SHIFT - loshift
  ret = value.digit(wordshift) >> loshift
  if newsize > 1:
    ret |= value.digit(wordshift+1) << hishift
  ret &= get_int_mask(masklen)
  return ret

_rbigint_rshift_maskoff_retint._always_inline_ = True

# This function implements getidx functionality.
@jit.elidable
def _rbigint_getidx( value, index ):
  wordpos = index / SHIFT
  if wordpos > value.numdigits(): return 0
  bitpos  = index - wordpos*SHIFT
  return (value.digit(wordpos) >> bitpos) & 1
_rbigint_getidx._always_inline_ = True

# This function implements setitem functionality.
@jit.elidable
def _rbigint_setidx( value, index, other ):
  size    = value.numdigits()
  wordpos = index / SHIFT

  if wordpos >= size:
    if not other:
      return value

    bitpos  = index - wordpos*SHIFT
    shift   = 1 << bitpos
    return rbigint( value._digits[:size] + \
                    [NULLDIGIT]*(wordpos-size) + \
                    [_store_digit(shift)], 1, wordpos + 1 )

  # wordpos < size
  digit = value.digit(wordpos)
  bitpos  = index - wordpos*SHIFT
  shift   = 1 << bitpos

  if other == 1:
    if digit & shift: # already 1
      return value
    # the value is changed
    ret = rbigint( value._digits[:size], 1, size )
    ret.setdigit( wordpos, digit | shift )
    return ret
  # other == 0

  if not (digit & shift): # already 0
    return value
  # the value is changed
  digit ^= shift
  if digit == 0 and wordpos == size-1:
    assert wordpos >= 0
    return rbigint( value._digits[:wordpos] + [NULLDIGIT], 1, wordpos )

  ret = rbigint(value._digits[:size], 1, size)
  ret.setdigit( wordpos, digit )
  return ret

# This function implements lshift+AND mask functionality.
# PLEASE NOTE THAT value should be from Bits.value, and masklen should
# be larger than this Bits object's nbits
@jit.elidable
def _rbigint_lshift_maskoff( value, shamt, masklen ):
  if not value.sign or not shamt: return value
  if shamt >= masklen:  return NULLRBIGINT
  assert shamt > 0
  # shamt must > 0, value.sign must >= 0

  wordshift = shamt // SHIFT
  remshift  = shamt - wordshift * SHIFT

  oldsize   = value.numdigits()

  maskbit   = masklen % SHIFT
  masksize  = (masklen - 1)/SHIFT + 1

  if not remshift:
    retsize = min( oldsize + wordshift, masksize )
    ret = rbigint( [NULLDIGIT]*retsize, 1, retsize )
    j = 0
    while j < oldsize and wordshift < retsize:
      ret.setdigit( wordshift, value.digit(j) )
      wordshift += 1
      j += 1

    if wordshift == masksize and maskbit != 0:
      lastword = retsize - 1
      ret.setdigit( lastword, ret.digit(lastword) & get_int_mask(maskbit) )

    ret._normalize()
    return ret

  newsize  = oldsize + wordshift + 1

  if masksize < newsize:
    retsize = masksize

    ret = rbigint([NULLDIGIT]*retsize, 1, retsize)
    accum = _widen_digit(0)
    j = 0
    while j < oldsize and wordshift < retsize:
      accum += value.widedigit(j) << remshift
      ret.setdigit(wordshift, accum)
      accum >>= SHIFT
      wordshift += 1
      j += 1

    # no accum
    if maskbit != 0:
      lastword  = retsize - 1
      lastdigit = ret.digit(lastword)
      mask      = get_int_mask(maskbit)
      if lastdigit >= mask:
        ret.setdigit( lastword, lastdigit & mask )

    ret._normalize()
    return ret

  # masksize >= newsize
  retsize = newsize

  ret = rbigint([NULLDIGIT]*retsize, 1, retsize)
  accum = _widen_digit(0)
  j = 0
  while j < oldsize and wordshift < retsize:
    accum += value.widedigit(j) << remshift
    ret.setdigit(wordshift, accum)
    accum >>= SHIFT
    wordshift += 1
    j += 1

  if masksize == newsize and maskbit != 0:
    accum &= get_int_mask(maskbit)

  ret.setdigit(wordshift, accum)

  ret._normalize()
  return ret

# setitem helpers

# Must return rbigint that cannot fit into int
@jit.elidable
def setitem_long_long_helper( value, other, start, stop ):
  if other.numdigits() <= 1:
    return setitem_long_int_helper( value, other.digit(0), start, stop )

  if other.sign < 0:
    slice_nbits = stop - start
    other = other.and_( get_long_mask(slice_nbits) )
    if other.numdigits() == 1:
      return setitem_long_int_helper( value, other.digit(0), start, stop )

  vsize = value.numdigits()
  other = other.lshift( start ) # lshift first to align two rbigints
  osize = other.numdigits()

  # Now other must be long, wordstart must < wordstop
  wordstart = start / SHIFT

  # vsize <= wordstart < wordstop, concatenate
  if wordstart >= vsize:
    return rbigint(value._digits[:vsize] + other._digits[vsize:], 1, osize )

  wordstop = stop / SHIFT # wordstop >= osize-1
  # (wordstart <) wordstop < vsize
  if wordstop < vsize:
    ret = rbigint( value._digits[:vsize], 1, vsize )

    # do start
    bitstart = start - wordstart*SHIFT
    tmpstart = other.digit( wordstart ) | (ret.digit(wordstart) & get_int_mask(bitstart))
    # if bitstart:
      # tmpstart |= ret.digit(wordstart) & get_int_mask(bitstart) # lo
    ret.setdigit( wordstart, tmpstart )

    i = wordstart+1

    if osize < wordstop:
      while i < osize:
        ret.setdigit( i, other.digit(i) )
        i += 1
      while i < wordstop:
        ret._digits[i] = NULLDIGIT
        i += 1
    else: # osize >= wordstop
      while i < wordstop:
        ret.setdigit( i, other.digit(i) )
        i += 1

    # do stop
    bitstop  = stop - wordstop*SHIFT
    if bitstop:
      masked_val = ret.digit(wordstop) & ~get_int_mask(bitstop) #hi
      ret.setdigit( wordstop, other.digit(wordstop) | masked_val ) # lo|hi

    return ret

  assert wordstart >= 0
  # wordstart < vsize <= wordstop
  ret = rbigint( value._digits[:wordstart] + \
                 other._digits[wordstart:osize], 1, osize )

  # do start
  bitstart = start - wordstart*SHIFT
  if bitstart:
    masked_val = value.digit(wordstart) & get_int_mask(bitstart) # lo
    ret.setdigit( wordstart, masked_val | ret.digit(wordstart) ) # lo | hi

  return ret

@jit.elidable
def setitem_long_int_helper( value, other, start, stop ):
  vsize = value.numdigits()
  if other < 0:
    slice_nbits = stop - start
    if slice_nbits < SHIFT:
      other &= get_int_mask(slice_nbits)
    else:
      tmp = get_long_mask(slice_nbits).int_and_( other )
      return setitem_long_long_helper( value, tmp, start, stop )

  # wordstart must < wordstop
  wordstart = start / SHIFT
  bitstart  = start - wordstart*SHIFT

  # vsize <= wordstart < wordstop, concatenate
  if wordstart >= vsize:
    if not other: return value # if other is zero, do nothing

    if not bitstart: # aha, not chopped into two parts
      return rbigint( value._digits[:vsize] + \
                      [NULLDIGIT]*(wordstart-vsize) + \
                      [_store_digit(other)], 1, wordstart+1 )

    # split into two parts
    lo = SHIFT-bitstart
    val1 = other & get_int_mask(lo)
    if val1 == other: # aha, the higher part is zero
      return rbigint( value._digits[:vsize] + \
                      [NULLDIGIT]*(wordstart-vsize) + \
                      [_store_digit(val1 << bitstart)], 1, wordstart+1 )
    return rbigint( value._digits[:vsize] + \
                    [NULLDIGIT]*(wordstart-vsize) + \
                    [_store_digit(val1 << bitstart)] + \
                    [_store_digit(other >> lo)], 1, wordstart+2 )

  wordstop = stop / SHIFT
  bitstop  = stop - wordstop*SHIFT
  # (wordstart <=) wordstop < vsize
  if wordstop < vsize:
    ret = rbigint( value._digits[:vsize], 1, vsize )
    maskstop = get_int_mask(bitstop)
    valstop  = ret.digit(wordstop)

    if wordstop == wordstart: # valstop is ret.digit(wordstart)
      valuemask = ~(maskstop - get_int_mask(bitstart))
      ret.setdigit( wordstop, (valstop & valuemask) | (other << bitstart) )

    # span multiple words
    # wordstart < wordstop
    else:
      # do start
      if not bitstart:
        ret.setdigit( wordstart, other )
        i = wordstart + 1
        while i < wordstop:
          ret._digits[i] = NULLDIGIT
          i += 1
        ret.setdigit( wordstop, valstop & ~maskstop )
      else:
        lo = SHIFT-bitstart
        val1 = other & get_int_mask(lo)
        word = (ret.digit(wordstart) & get_int_mask(bitstart)) | (val1 << bitstart)
        ret.setdigit( wordstart, word )

        val2 = other >> lo
        i = wordstart + 1
        if i == wordstop:
          ret.setdigit( i, val2 | (valstop & ~maskstop) )
        else: # i < wordstop
          ret.setdigit( i, val2 )
          i += 1
          while i < wordstop:
            ret._digits[i] = NULLDIGIT
            i += 1
          ret.setdigit( wordstop, valstop & ~maskstop )
    ret._normalize()
    return ret

  # wordstart < vsize <= wordstop, highest bits will be cleared
  newsize = wordstart + 2 #
  assert wordstart >= 0
  ret = rbigint( value._digits[:wordstart] + \
                [NULLDIGIT, NULLDIGIT], 1, newsize )

  bitstart = start - wordstart*SHIFT
  if not bitstart:
    ret.setdigit( wordstart, other )
  else:
    lo = SHIFT-bitstart
    val1 = other & get_int_mask(lo)
    word = (value.digit(wordstart) & get_int_mask(bitstart)) | (val1 << bitstart)
    ret.setdigit( wordstart, word )

    if val1 != other:
      ret.setdigit( wordstart+1, other >> lo )

  ret._normalize()
  return ret

setitem_long_int_helper._always_inline_ = True

class W_BigBits(W_AbstractBits):
  __slots__ = ( "nbits", "bigval" )
  _immutable_fields_ = [ "nbits" ]

  def __init__( self, nbits, bigval ):
    self.nbits  = nbits
    self.bigval = bigval

  def descr_copy(self):
    return W_BigBits( self.nbits, self.bigval )

  #-----------------------------------------------------------------------
  # get/setitem
  #-----------------------------------------------------------------------

  def descr_getitem(self, space, w_index):
    if type(w_index) is W_SliceObject:
      if space.is_w(w_index.w_step, space.w_None):
        start, stop = _get_slice_range( space, self.nbits, w_index.w_start, w_index.w_stop )

        slice_nbits = stop - start
        if slice_nbits <= SHIFT:
          return W_SmallBits( slice_nbits, _rbigint_rshift_maskoff_retint( self.bigval, start, slice_nbits ) )
        else:
          return W_BigBits( slice_nbits, _rbigint_rshift_maskoff( self.bigval, start, slice_nbits ) )

      else:
        raise oefmt(space.w_ValueError, "Bits slice cannot have step." )

    else:
      index = _get_index(space, self.nbits, w_index)
      return W_SmallBits( 1, _rbigint_getidx( self.bigval, index ) )

  def descr_setitem(self, space, w_index, w_other):
    if type(w_index) is W_SliceObject:
      if space.is_w(w_index.w_step, space.w_None):
        start, stop = _get_slice_range( space, self.nbits, w_index.w_start, w_index.w_stop )

        slice_nbits = stop - start

        if isinstance(w_other, W_SmallBits):
          self.bigval = setitem_long_int_helper( self.bigval, w_other.intval, start, stop )

        elif isinstance(w_other, W_IntObject):
          other = w_other.intval
          blen  = int_bit_length( other )
          if blen > slice_nbits:
            raise oefmt(space.w_ValueError, "Value %d cannot fit into "
                  "[%d:%d] (%d-bit) slice", other, start, stop, slice_nbits )
          other = get_long_mask(slice_nbits).int_and_( other )
          self.bigval = setitem_long_long_helper( self.bigval, other, start, stop )

        elif isinstance(w_other, W_BigBits):
          self.bigval = setitem_long_long_helper( self.bigval, w_other.bigval, start, stop )

        elif type(w_other) is W_LongObject:
          other = w_other.num
          blen = other.bit_length()
          if blen > slice_nbits:
            raise oefmt(space.w_ValueError, "Value %s cannot fit into "
                  "[%d:%d] (%d-bit) slice", rbigint.str(other), start, stop, slice_nbits )

          other = get_long_mask(slice_nbits).and_( other )
          self.bigval = setitem_long_long_helper( self.bigval, other, start, stop )

      else:
        raise oefmt(space.w_ValueError, "Bits slice cannot have step." )

    else:
      index = _get_index(space, self.nbits, w_index)

      # Check value bitlen. No need to check Bits, but check int/long.
      if isinstance(w_other, W_SmallBits):
        o_nbits = w_other.nbits
        if o_nbits > 1:
          raise oefmt(space.w_ValueError, "Bits%d cannot fit into 1-bit slice", o_nbits )
        other = w_other.intval # must be 1-bit and don't even check

        self.bigval = _rbigint_setidx( self.bigval, index, other )

      elif isinstance(w_other, W_IntObject):
        other = w_other.intval
        if other < 0 or other > 1:
          raise oefmt(space.w_ValueError, "Value %d cannot fit into 1-bit slice", other )

        self.bigval = _rbigint_setidx( self.bigval, index, other )

      elif isinstance(w_other, W_BigBits):
        raise oefmt(space.w_ValueError, "Bits%d cannot fit into 1-bit slice", w_other.nbits )

      elif type(w_other) is W_LongObject:
        other = w_other.num
        if other.numdigits() > 1:
          raise oefmt(space.w_ValueError, "Value %s cannot fit into 1-bit slice", rbigint.str(other) )

        other = other.digit(0)
        if other < 0 or other > 1:
          raise oefmt(space.w_ValueError, "Value %d cannot fit into 1-bit slice", other )

        self.bigval = _rbigint_setidx( self.bigval, index, other )
      else:
        raise oefmt(space.w_TypeError, "Please pass in int/long/Bits value. -- setitem #4" )

  #-----------------------------------------------------------------------
  # Miscellaneous methods for string format
  #-----------------------------------------------------------------------

  def _format16(self, space):
    data = self.bigval.format(BASE16)
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
    llop = getattr( rbigint , opname )
    liop = getattr( rbigint , "int_"+opname )

    @func_renamer('descr_' + opname)
    def descr_cmp(self, space, w_other):
      x = self.bigval

      if isinstance(w_other, W_BigBits):
        return W_SmallBits( 1, llop( x, w_other.bigval ) )

      elif isinstance(w_other, W_SmallBits):
        return W_SmallBits( 1, liop( x, w_other.intval ) )

      elif isinstance(w_other, W_IntObject):
        # TODO Maybe add bit_length check?
        return W_SmallBits( 1, llop( x, get_long_mask(self.nbits).int_and_( w_other.intval ) ) )

      elif type(w_other) is W_LongObject:
        # TODO Maybe add bit_length check?
        return W_SmallBits( 1, llop( x, get_long_mask(self.nbits).and_( w_other.num ) ) )

      return W_SmallBits( 1, 0 )
      # Match cpython behavior
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

    @func_renamer('descr_' + opname)
    def descr_binop(self, space, w_other):
      # add, sub, mul
      x = self.bigval

      if ovf:
        if isinstance(w_other, W_BigBits):
          z = llop( x, w_other.bigval )
          res_nbits = max(self.nbits, w_other.nbits)
          if opname == "sub": z = z.and_( get_long_mask(res_nbits) )
          else:               z = _rbigint_maskoff_high( z, res_nbits )
          return W_BigBits( res_nbits, z )

        elif isinstance(w_other, W_SmallBits):
          z = liop( x, w_other.intval )
          if opname == "sub": z = z.and_( get_long_mask(self.nbits) )
          else:               z = _rbigint_maskoff_high( z, self.nbits )
          return W_BigBits( self.nbits, z )

        elif type(w_other) is W_IntObject:
          z = liop( x, w_other.intval )
          if opname == "sub": z = z.and_( get_long_mask(self.nbits) )
          else:               z = _rbigint_maskoff_high( z, self.nbits )
          return W_BigBits( self.nbits, z )

        elif type(w_other) is W_LongObject:
          z = llop( x, w_other.num )
          if opname == "sub": z = z.and_( get_long_mask(self.nbits) )
          else:               z = _rbigint_maskoff_high( z, self.nbits )
          return W_BigBits( self.nbits, z )

      # and, or, xor, no overflow
      # opname should be in COMMUTATIVE_OPS
      else:
        if isinstance(w_other, W_SmallBits):
          return W_BigBits( self.nbits, liop( x, w_other.intval ) )
        elif isinstance(w_other, W_BigBits):
          return W_BigBits( max(self.nbits, w_other.nbits), llop( x, w_other.bigval ) )
        elif isinstance(w_other, W_IntObject):
          # TODO Maybe add int_bit_length check?
          return W_BigBits( self.nbits, liop( x, w_other.intval ) )
        elif type(w_other) is W_LongObject:
          # TODO Maybe add int_bit_length check?
          return W_BigBits( self.nbits, llop( x, w_other.num ) )

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

    y = self.bigval
    if isinstance(w_other, W_IntObject):
      z = llop( rbigint.fromint(w_other.intval), y )
      z = z.and_( get_long_mask(self.nbits) )
      return W_BigBits( self.nbits, z )

    elif type(w_other) is W_LongObject:
      z = llop( w_other.num, y )
      z = z.and_( get_long_mask(self.nbits) )
      return W_BigBits( self.nbits, z )

  descr_add, descr_radd = _make_descr_binop_opname('add')
  descr_sub, _          = _make_descr_binop_opname('sub')
  descr_mul, descr_rmul = _make_descr_binop_opname('mul')

  descr_and, descr_rand = _make_descr_binop_opname('and', ovf=False)
  descr_or, descr_ror   = _make_descr_binop_opname('or', ovf=False)
  descr_xor, descr_rxor = _make_descr_binop_opname('xor', ovf=False)

  def descr_rshift(self, space, w_other):

    x = self.bigval

    if isinstance(w_other, W_SmallBits):
      return W_BigBits( self.nbits, x.rshift( w_other.intval ) )

    elif isinstance(w_other, W_IntObject):
      return W_BigBits( self.nbits, x.rshift( w_other.intval ) )

    elif isinstance(w_other, W_BigBits):
      return W_BigBits( self.nbits, _rbigint_rshift( x, w_other.bigval ) )

    elif type(w_other) is W_LongObject:
      return W_BigBits( self.nbits, _rbigint_rshift( x, w_other.num ) )

    raise oefmt(space.w_TypeError, "Please do rshift between <Bits, Bits/int/long> objects" )

  def descr_lshift(self, space, w_other):

    x = self.bigval

    if isinstance(w_other, W_SmallBits):
      shamt = w_other.intval
      return W_BigBits( self.nbits, _rbigint_lshift_maskoff( x, shamt, self.nbits ) )

    elif type(w_other) is W_IntObject:
      return W_BigBits( self.nbits, _rbigint_lshift_maskoff( x, w_other.intval, self.nbits ) )

    elif isinstance(w_other, W_BigBits):
      shamt = w_other.bigval
      if shamt.numdigits() > 1: return W_BigBits( self.nbits, NULLRBIGINT ) # rare
      shamt = shamt.digit(0)
      return W_BigBits( self.nbits, _rbigint_lshift_maskoff( x, shamt, self.nbits ) )

    elif type(w_other) is W_LongObject:
      shamt = w_other.num
      if shamt.numdigits() > 1: return W_BigBits( self.nbits, NULLRBIGINT ) # rare
      shamt = shamt.digit(0)
      return W_BigBits( self.nbits, _rbigint_lshift_maskoff( x, shamt, self.nbits ) )

    raise oefmt(space.w_TypeError, "Please do lshift between <Bits, Bits/int/long> objects" )

  def descr_rlshift(self, space, w_other): # int << Bits, what is nbits??
    raise oefmt(space.w_TypeError, "rlshift not implemented" )

  #-----------------------------------------------------------------------
  # <<=
  #-----------------------------------------------------------------------

  def _descr_ilshift(self, space, w_other):

    if isinstance(w_other, W_BigBits):
      if self.nbits != w_other.nbits:
        raise oefmt(space.w_ValueError, "Bitwidth mismatch Bits%d <> Bits%d",
                                        self.nbits, w_other.nbits)
      return W_BigBitsWithNext( self.nbits, self.bigval, w_other.bigval )

    elif isinstance(w_other, W_SmallBits):
      raise oefmt(space.w_ValueError, "Bitwidth mismatch during <<=, assigning Bits%d <<= Bits%d",
                                      self.nbits, w_other.nbits)
    else:
      raise oefmt(space.w_TypeError, "RHS of <<= has to be Bits%d, not '%T'", self.nbits, w_other)

  def _descr_flip(self, space):
    raise oefmt(space.w_TypeError, "_flip cannot be called on '%T' objects which has no _next", self)

  #-----------------------------------------------------------------------
  # value access
  #-----------------------------------------------------------------------

  def descr_uint(self, space):
    return newlong( space, self.bigval )

  def descr_int(self, space): # TODO
    index = self.nbits - 1
    bigval = self.bigval
    wordpos = index / SHIFT
    if wordpos > bigval.numdigits(): # msb must be zero, number is positive
      return self.descr_uint( space )

    bitpos = index - wordpos*SHIFT
    word = bigval.digit( wordpos )
    msb = (word >> bitpos) & 1
    if not msb:
      return newlong( space, bigval )

    # calculate self.nbits's index
    bitpos += 1
    if bitpos == SHIFT:
      wordpos += 1
      bitpos = 0

    # manually assemble (1 << (index+1))
    shift = rbigint( [NULLDIGIT]*wordpos + [_store_digit(1 << bitpos)],
                     1, wordpos+1 )

    res = bigval.sub( shift )
    return newlong( space, res )

  descr_pos = func_with_new_name( descr_uint, 'descr_pos' )
  descr_index = func_with_new_name( descr_uint, 'descr_index' )

  #-----------------------------------------------------------------------
  # unary ops
  #-----------------------------------------------------------------------

  def descr_bool(self, space):
    return space.newbool( self.bigval.sign != 0 )

  def descr_invert(self, space):
    return W_BigBits( self.nbits, get_long_mask(self.nbits).sub( self.bigval ) )

  # def descr_neg(self, space):

  def descr_hash(self, space):
    hash_nbits = _hash_int( self.nbits )
    hash_value = _hash_long( space, self.bigval )

    # Manually implement a single iter of W_TupleObject.descr_hash

    x = 0x345678
    x = (x ^ hash_nbits) * 1000003
    x = (x ^ hash_value) * (1000003+82520+1+1)
    x += 97531
    return space.newint( intmask(x) )

#-----------------------------------------------------------------------
# Bits with next fields
#-----------------------------------------------------------------------

class W_BigBitsWithNext(W_BigBits):
  __slots__ = ( "nbits", "bigval", "next_bigval" )
  _immutable_fields_ = [ "nbits" ]

  def __init__( self, nbits, bigval, next_bigval ):
    self.nbits  = nbits
    self.bigval = bigval
    self.next_bigval = next_bigval

  # def descr_setitem(self, space, w_index, w_other):
    # raise oefmt(space.w_TypeError, "You shouldn't do x[a:b]=y on flip-flop")

  def descr_copy(self):
    return W_BigBitsWithNext( self.nbits, self.bigval, self.next_bigval )

  def _descr_ilshift(self, space, w_other):

    if isinstance(w_other, W_BigBits):
      if self.nbits != w_other.nbits:
        raise oefmt(space.w_ValueError, "Bitwidth mismatch during <<=, assigning Bits%d <<= Bits%d",
                                        self.nbits, w_other.nbits)
      self.next_bigval = w_other.bigval
    elif isinstance(w_other, W_SmallBits):
      raise oefmt(space.w_ValueError, "Bitwidth mismatch during <<=, assigning Bits%d <<= Bits%d",
                                      self.nbits, w_other.nbits)
    return self

  def _descr_flip(self, space):
    self.bigval = self.next_bigval
