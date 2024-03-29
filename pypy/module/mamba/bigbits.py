import operator

from rpython.rlib.rarithmetic import intmask
from rpython.rlib.rbigint     import rbigint, _store_digit, NULLDIGIT, NULLRBIGINT, SHIFT, BASE8, BASE16
from rpython.tool.sourcetools import func_renamer, func_with_new_name

from pypy.module.mamba.smallbits import W_AbstractBits, W_SmallBits, \
                                        cmp_opp, _get_index, _get_slice_range

from pypy.interpreter.baseobjspace import W_Root
from pypy.interpreter.gateway import WrappedDefault, interp2app, interpindirect2app, unwrap_spec
from pypy.interpreter.error import OperationError, oefmt
from pypy.objspace.std.intobject import W_IntObject, wrapint, ovfcheck, _hash_int
from pypy.objspace.std.longobject import W_LongObject, newlong, _hash_long
from pypy.objspace.std.sliceobject import W_SliceObject
from pypy.objspace.std.util import COMMUTATIVE_OPS

from pypy.module.mamba.helper_funcs import BASE2, get_int_mask, get_long_mask, get_int_lower, get_long_lower, \
  _rbigint_check_exceed_nbits, _rbigint_invalid_binop_operand, _rbigint_maskoff_high, \
  _rbigint_rshift, _rbigint_rshift_maskoff, _rbigint_rshift_maskoff_retint, _rbigint_getidx, \
  _rbigint_setidx, _rbigint_lshift_maskoff, setitem_long_long_helper, setitem_long_int_helper

# NOTE that we should keep self.value positive after any computation:
# - The sign of the rbigint field should always be one
# - Always AND integer value with mask, never store any negative int
# * Performing rbigint.and_/rbigint.int_and_ will turn sign back to 1
# - rbigint._normalize() can only be called in @jit.elidable funcs

class W_BigBits(W_AbstractBits):
  __slots__ = ( "nbits", "bigval" )
  _immutable_fields_ = [ "nbits" ]

  def __init__( self, nbits, bigval ):
    self.nbits  = nbits
    self.bigval = bigval

  def _descr_copy(self):
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
        raise oefmt(space.w_IndexError, "Index cannot contain step" )
    else:
      index = _get_index(space, self.nbits, w_index)
      return W_SmallBits( 1, _rbigint_getidx( self.bigval, index ) )

  def descr_setitem(self, space, w_index, w_other):
    if type(w_index) is W_SliceObject:
      if space.is_w(w_index.w_step, space.w_None):
        start, stop = _get_slice_range( space, self.nbits, w_index.w_start, w_index.w_stop )
        slice_nbits = stop - start

        if isinstance(w_other, W_SmallBits):
          if w_other.nbits != slice_nbits:
            if w_other.nbits < slice_nbits:
              raise oefmt(space.w_ValueError, "Cannot fit a Bits%d object into a %d-bit slice [%d:%d]\n"
                                "- Suggestion: sext/zext the RHS", w_other.nbits, slice_nbits, start, stop)
            else:
              raise oefmt(space.w_ValueError, "Cannot fit a Bits%d object into a %d-bit slice [%d:%d]\n"
                                "- Suggestion: trunc the RHS", w_other.nbits, slice_nbits, start, stop)

          self.bigval = setitem_long_int_helper( self.bigval, w_other.intval, start, stop )

        elif isinstance(w_other, W_IntObject):
          other = w_other.intval
          up = get_int_mask(slice_nbits)
          lo = get_int_lower(slice_nbits)

          if other < lo or other > up:
            raise oefmt(space.w_ValueError, "Cannot fit value %s into a Bits%d slice!\n"
                                            "(Bits%d only accepts %s <= value <= %s)",
                                            hex(other), slice_nbits, slice_nbits, hex(lo), hex(up))
          if slice_nbits < SHIFT:
            other = other & get_int_mask(slice_nbits)
            self.bigval = setitem_long_int_helper( self.bigval, other, start, stop )
          else:
            other = get_long_mask(slice_nbits).int_and_(other)
            self.bigval = setitem_long_long_helper( self.bigval, other, start, stop )

        elif isinstance(w_other, W_BigBits):
          if w_other.nbits != slice_nbits:
            if w_other.nbits < slice_nbits:
              raise oefmt(space.w_ValueError, "Cannot fit a Bits%d object into a %d-bit slice [%d:%d]\n"
                                "- Suggestion: sext/zext the RHS", w_other.nbits, slice_nbits, start, stop)
            else:
              raise oefmt(space.w_ValueError, "Cannot fit a Bits%d object into a %d-bit slice [%d:%d]\n"
                                "- Suggestion: trunc the RHS", w_other.nbits, slice_nbits, start, stop)

          self.bigval = setitem_long_long_helper( self.bigval, w_other.bigval, start, stop )

        elif isinstance(w_other, W_LongObject):
          other = w_other.num
          if _rbigint_check_exceed_nbits( other, slice_nbits ):
            raise oefmt(space.w_ValueError, "Cannot fit value %s into a Bits%d slice!\n"
                                            "(Bits%d only accepts %s <= value <= %s)",
                                            other.format(BASE16, prefix='0x'), slice_nbits, slice_nbits,
                                            hex(get_int_lower(slice_nbits)),hex(get_int_mask(slice_nbits)))
          if slice_nbits < SHIFT:
            other = other.int_and_(get_int_mask(slice_nbits)).digit(0)
            self.bigval = setitem_long_int_helper( self.bigval, other, start, stop )

          else:
            other = other.and_( get_long_mask(slice_nbits) )
            self.bigval = setitem_long_long_helper( self.bigval, other, start, stop )

      else:
        raise oefmt(space.w_IndexError, "Index cannot contain step" )

    else:
      index = _get_index(space, self.nbits, w_index)

      # Check value bitlen. No need to check Bits, but check int/long.
      if isinstance(w_other, W_SmallBits):
        o_nbits = w_other.nbits
        if o_nbits > 1:
          raise oefmt(space.w_ValueError, "Cannot fit a Bits%d object into an 1-bit slice", o_nbits )
        other = w_other.intval # must be 1-bit and don't even check

        self.bigval = _rbigint_setidx( self.bigval, index, other )

      elif isinstance(w_other, W_IntObject):
        other = w_other.intval
        if other < -1 or other > 1:
          raise oefmt(space.w_ValueError, "Value %s is too big for the 1-bit slice", hex(other) )

        self.bigval = _rbigint_setidx( self.bigval, index, other & 1 )

      elif isinstance(w_other, W_BigBits):
        raise oefmt(space.w_ValueError, "Cannot fit a Bits%d object into 1-bit slice", w_other.nbits )

      elif type(w_other) is W_LongObject:
        other = w_other.num
        if other.numdigits() > 1:
          raise oefmt(space.w_ValueError, "Value %s is too big for the 1-bit slice", other.format(BASE16, prefix='0x') )
        lsw = other.digit(0)
        if lsw > 1: # -1 and 1 are both 1 here
          raise oefmt(space.w_ValueError, "Value %s is too big for the 1-bit slice", other.format(BASE16, prefix='0x') )
        self.bigval = _rbigint_setidx( self.bigval, index, lsw )
      else:
        raise oefmt(space.w_TypeError, "Please pass in int/long/Bits value. -- setitem #4" )

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
        if self.nbits != w_other.nbits:
          raise oefmt(space.w_ValueError, "Operands of '%s' operation must have matching bitwidth, "
                                          "but here Bits%d != Bits%d.\n", opname, self.nbits, w_other.nbits )
        return W_SmallBits( 1, llop( x, w_other.bigval ) )

      elif isinstance(w_other, W_SmallBits):
        raise oefmt(space.w_ValueError, "Operands of '%s' operation must have matching bitwidth, "
                                        "but here Bits%d != Bits%d.\n", opname, self.nbits, w_other.nbits )

      elif isinstance(w_other, W_IntObject): # int MUST fit Bits64+
        return W_SmallBits( 1, llop( x, get_long_mask(self.nbits).int_and_( w_other.intval ) ) )

      elif type(w_other) is W_LongObject:
        nbits = self.nbits
        y = w_other.num

        if _rbigint_invalid_binop_operand( y, nbits ):
          raise oefmt(space.w_ValueError, "Integer %s is not a valid binop operand with Bits%d!\n"
                                          "Suggestion: 0 <= x <= %s", y.format(BASE16, prefix='0x'), nbits,
                                          get_long_mask(nbits).format(BASE16, prefix='0x'))

        return W_SmallBits( 1, llop( x, get_long_mask(nbits).and_( w_other.num ) ) )

      if opname == 'eq':
        # Match cpython behavior
        return W_SmallBits( 1, 0 )
      elif opname == 'ne':
        # Match cpython behavior
        return W_SmallBits( 1, 1 )

      raise oefmt(space.w_TypeError, "Please compare two Bits/int/long objects" )

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
      nbits = self.nbits

      if ovf:
        if isinstance(w_other, W_BigBits):
          if nbits != w_other.nbits:
            raise oefmt(space.w_ValueError, "Operands of '%s' operation must have matching bitwidth, "
                                            "but here Bits%d != Bits%d.\n", opname, nbits, w_other.nbits )
          z = llop( x, w_other.bigval )
          if opname == "sub": z = z.and_( get_long_mask(nbits) )
          else:               z = _rbigint_maskoff_high( z, nbits )
          return W_BigBits( nbits, z )

        elif isinstance(w_other, W_IntObject): # int MUST fit Bits64+
          z = liop( x, w_other.intval )
          if opname == "sub": z = z.and_( get_long_mask(nbits) )
          else:               z = _rbigint_maskoff_high( z, nbits )
          return W_BigBits( nbits, z )

        elif isinstance(w_other, W_LongObject):
          y = w_other.num
          if _rbigint_invalid_binop_operand( y, nbits ):
            raise oefmt(space.w_ValueError, "Integer %s is not a valid binop operand with Bits%d!\n"
                                            "Suggestion: 0 <= x <= %s", y.format(BASE16, prefix='0x'), nbits,
                                            get_long_mask(nbits).format(BASE16, prefix='0x'))

          z = llop( x, y )
          if opname == "sub": z = z.and_( get_long_mask(nbits) )
          else:               z = _rbigint_maskoff_high( z, nbits )
          return W_BigBits( nbits, z )

        elif isinstance(w_other, W_SmallBits):
          raise oefmt(space.w_ValueError, "Operands of '%s' operation must have matching bitwidth, "
                                          "but here Bits%d != Bits%d.\n", opname, nbits, w_other.nbits )

      # and, or, xor, no overflow
      # opname should be in COMMUTATIVE_OPS
      else:
        if   isinstance(w_other, W_BigBits):
          if nbits != w_other.nbits:
            raise oefmt(space.w_ValueError, "Operands of '%s' operation must have matching bitwidth, "
                                            "but here Bits%d != Bits%d.\n", opname, nbits, w_other.nbits )
          return W_BigBits( nbits, llop( x, w_other.bigval ) )

        elif isinstance(w_other, W_IntObject): # int MUST fit Bits64+
          return W_BigBits( nbits, liop( x, w_other.intval ) )

        elif isinstance(w_other, W_LongObject):
          y = w_other.num
          if _rbigint_invalid_binop_operand( y, nbits ):
            raise oefmt(space.w_ValueError, "Integer %s is not a valid binop operand with Bits%d!\n"
                                            "Suggestion: 0 <= x <= %s", y.format(BASE16, prefix='0x'), nbits,
                                            get_long_mask(nbits).format(BASE16, prefix='0x'))
          return W_BigBits( nbits, llop( x, y ) )

        elif isinstance(w_other, W_SmallBits):
          raise oefmt(space.w_ValueError, "Operands of '%s' operation must have matching bitwidth, "
                                          "but here Bits%d != Bits%d.\n", opname, nbits, w_other.nbits )
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
  def _make_descr_rbinop_opname(opname):
    llop = getattr( rbigint, opname )

    @func_renamer('descr_r' + opname)
    def descr_rbinop(self, space, w_other):
      y = self.bigval
      nbits = self.nbits

      if isinstance(w_other, W_IntObject):  # int MUST fit Bits64+
        z = llop( rbigint.fromint(w_other.intval), y )
        z = z.and_( get_long_mask(nbits) )
        return W_BigBits( nbits, z )

      elif type(w_other) is W_LongObject:
        x = w_other.num
        if _rbigint_invalid_binop_operand( x, nbits ):
          raise oefmt(space.w_ValueError, "Integer %s is not a valid binop operand with Bits%d!\n"
                                          "Suggestion: 0 <= x <= %s", x.format(BASE16, prefix='0x'), nbits,
                                          get_long_mask(nbits).format(BASE16, prefix='0x'))
        z = llop( x, y )
        return W_BigBits( nbits, z.and_( get_long_mask(nbits) ) )
    return descr_rbinop

  descr_add, descr_radd = _make_descr_binop_opname('add')
  descr_mul, descr_rmul = _make_descr_binop_opname('mul')
  descr_sub, _          = _make_descr_binop_opname('sub')
  descr_floordiv, _     = _make_descr_binop_opname('floordiv')
  descr_mod, _          = _make_descr_binop_opname('mod')

  descr_and, descr_rand = _make_descr_binop_opname('and', ovf=False)
  descr_or, descr_ror   = _make_descr_binop_opname('or', ovf=False)
  descr_xor, descr_rxor = _make_descr_binop_opname('xor', ovf=False)

  descr_rsub      = _make_descr_rbinop_opname('sub')
  descr_rfloordiv = _make_descr_rbinop_opname('floordiv')
  descr_rmod      = _make_descr_rbinop_opname('mod')

  def descr_rshift(self, space, w_other):

    x = self.bigval

    if isinstance(w_other, W_IntObject):
      shamt = w_other.intval
      if shamt < 0: # int must be in bigbits range
        raise oefmt(space.w_ValueError, "Integer %s is not a valid binop operand with Bits%d!\n"
                                        "Suggestion: 0 <= x <= %s", hex(shamt), self.nbits,
                                        get_long_mask(self.nbits).format(BASE16, prefix='0x'))
      return W_BigBits( self.nbits, x.rshift( shamt ) )

    elif isinstance(w_other, W_BigBits):
      if self.nbits != w_other.nbits:
        raise oefmt(space.w_ValueError, "Operands of '>>' (rshift) operation must have matching bitwidth, "
                                        "but here Bits%d != Bits%d.\n", self.nbits, w_other.nbits )
      return W_BigBits( self.nbits, _rbigint_rshift( x, w_other.bigval ) )

    elif type(w_other) is W_LongObject:
      big = w_other.num
      if _rbigint_invalid_binop_operand( big, self.nbits ):
        raise oefmt(space.w_ValueError, "Integer %s is not a valid binop operand with Bits%d!\n"
                                        "Suggestion: 0 <= x <= %s", big.format(BASE16, prefix='0x'), self.nbits,
                                        get_long_mask(self.nbits).format(BASE16, prefix='0x'))
      return W_BigBits( self.nbits, _rbigint_rshift( x, big ) )

    elif isinstance(w_other, W_SmallBits):
      raise oefmt(space.w_ValueError, "Operands of '>>' (rshift) operation must have matching bitwidth, "
                                      "but here Bits%d != Bits%d.\n", self.nbits, w_other.nbits )

    raise oefmt(space.w_TypeError, "Please do rshift between <Bits, Bits/int/long> objects" )

  def descr_lshift(self, space, w_other):

    x = self.bigval

    if isinstance(w_other, W_IntObject):
      shamt = w_other.intval
      if shamt < 0: # int must be in bigbits range
        raise oefmt(space.w_ValueError, "Integer %s is not a valid binop operand with Bits%d!\n"
                                        "Suggestion: 0 <= x <= %s", hex(shamt), self.nbits,
                                        get_long_mask(self.nbits).format(BASE16, prefix='0x'))
      return W_BigBits( self.nbits, _rbigint_lshift_maskoff( x, shamt, self.nbits ) )

    elif isinstance(w_other, W_BigBits):
      if self.nbits != w_other.nbits:
        raise oefmt(space.w_ValueError, "Operands of '<<' (lshift) operation must have matching bitwidth, "
                                        "but here Bits%d != Bits%d.\n", self.nbits, w_other.nbits )
      shamt = w_other.bigval
      if shamt.numdigits() > 1: return W_BigBits( self.nbits, NULLRBIGINT ) # rare
      shamt = shamt.digit(0)
      return W_BigBits( self.nbits, _rbigint_lshift_maskoff( x, shamt, self.nbits ) )

    elif type(w_other) is W_LongObject:
      shamt = w_other.num
      if _rbigint_invalid_binop_operand( shamt, self.nbits ):
        raise oefmt(space.w_ValueError, "Integer %s is not a valid binop operand with Bits%d!\n"
                                        "Suggestion: 0 <= x <= %s", shamt.format(BASE16, prefix='0x'), self.nbits,
                                        get_long_mask(self.nbits).format(BASE16, prefix='0x'))
      if shamt.numdigits() > 1: return W_BigBits( self.nbits, NULLRBIGINT ) # rare
      shamt = shamt.digit(0)
      return W_BigBits( self.nbits, _rbigint_lshift_maskoff( x, shamt, self.nbits ) )

    elif isinstance(w_other, W_SmallBits):
      raise oefmt(space.w_ValueError, "Operands of '<<' (lshift) operation must have matching bitwidth, "
                                      "but here Bits%d != Bits%d.\n", self.nbits, w_other.nbits )

    raise oefmt(space.w_TypeError, "Please do lshift between <Bits, Bits/int/long> objects" )

  def descr_rlshift(self, space, w_other): # int << Bits, what is nbits??
    raise oefmt(space.w_TypeError, "rlshift not implemented" )

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

  # PyMTL specific
  #        |
  #        |
  #        V

  #-----------------------------------------------------------------------
  # @=
  #-----------------------------------------------------------------------

  def _descr_imatmul(self, space, w_other):
    nbits = self.nbits

    if isinstance(w_other, W_BigBits):
      if nbits != w_other.nbits:
        if nbits > w_other.nbits:
          raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during @= blocking assignment, "
                                          "but here LHS Bits%d > RHS Bits%d.\n"
                                          "- Suggestion: LHS @= zext/sext(RHS, nbits/Type)", nbits, w_other.nbits)
        else:
          raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during @= blocking assignment, "
                                          "but here LHS Bits%d < RHS Bits%d.\n"
                                          "- Suggestion: LHS @= trunc(RHS, nbits/Type)", nbits, w_other.nbits)
      self.bigval = w_other.bigval

    elif isinstance(w_other, W_IntObject): # int MUST fit Bits64+
      self.bigval = get_long_mask(nbits).int_and_(w_other.intval)

    elif isinstance(w_other, W_LongObject):
      bigval = w_other.num
      if _rbigint_check_exceed_nbits( bigval, nbits ):
        raise oefmt(space.w_ValueError, "RHS value %s of @= is too wide for LHS Bits%d!\n" \
                                        "(Bits%d only accepts %s <= value <= %s)",
                                        bigval.format(BASE16, prefix='0x'), nbits, nbits,
                                        hex(get_int_lower(nbits)), hex(get_int_mask(nbits)))
      self.bigval = get_long_mask(self.nbits).and_(bigval)

    elif isinstance(w_other, W_SmallBits):
      raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during @= blocking assignment, "
                                      "but here LHS Bits%d > RHS Bits%d.\n"
                                      "- Suggestion: LHS @= sext/zext(RHS, nbits/Type)", self.nbits, w_other.nbits )

    else:
      w_other_bits = space.call_method(w_other, 'to_bits')
      if isinstance(w_other_bits, W_BigBits):
        if nbits != w_other_bits.nbits:
          if nbits > w_other_bits.nbits:
            raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS to_bits() during @= blocking assignment, "
                                            "but here LHS Bits%d > RHS Bits%d.\n"
                                            "- Suggestion: LHS @= zext/sext(RHS, nbits/Type)", nbits, w_other_bits.nbits)
          else:
            raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during @= blocking assignment, "
                                            "but here LHS Bits%d < RHS Bits%d.\n"
                                            "- Suggestion: LHS @= zext/sext(RHS, nbits/Type)", nbits, w_other_bits.nbits)
        self.bigval = w_other_bits.bigval

      elif isinstance(w_other_bits, W_SmallBits):
        raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS to_bits() during @= blocking assignment, "
                                        "but here LHS Bits%d > RHS Bits%d.\n"
                                        "- Suggestion: LHS @= zext/sext(RHS, nbits/Type)", nbits, w_other_bits.nbits )
      else:
        raise oefmt(space.w_ValueError, "RHS to_bits() must return Bits object!")

    return self

  #-----------------------------------------------------------------------
  # <<=
  #-----------------------------------------------------------------------

  def _descr_ilshift(self, space, w_other):
    nbits = self.nbits

    if isinstance(w_other, W_BigBits):
      if nbits != w_other.nbits:
        if nbits > w_other.nbits:
          raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during <<= non-blocking assignment, "
                                          "but here LHS Bits%d > RHS Bits%d.\n"
                                          "- Suggestion: LHS <<= zext/sext(RHS, nbits/Type)", nbits, w_other.nbits)
        else:
          raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during <<= non-blocking assignment, "
                                          "but here LHS Bits%d < RHS Bits%d.\n"
                                          "- Suggestion: LHS <<= trunc(RHS, nbits/Type)", nbits, w_other.nbits)
      return W_BigBitsWithNext( nbits, self.bigval, w_other.bigval )

    elif isinstance(w_other, W_IntObject): # int MUST fit Bits64+
      return W_BigBitsWithNext( nbits, self.bigval, get_long_mask(nbits).int_and_(w_other.intval) )

    elif isinstance(w_other, W_LongObject):
      bigval = w_other.num
      if _rbigint_check_exceed_nbits( bigval, nbits ):
        raise oefmt(space.w_ValueError, "RHS value %s of <<= is too wide for LHS Bits%d!\n"
                                        "(Bits%d only accepts %s <= value <= %s)",
                                        bigval.format(BASE16, prefix='0x'), nbits, nbits,
                                        hex(get_int_lower(nbits)), hex(get_int_mask(nbits)))
      return W_BigBitsWithNext( nbits, self.bigval, get_long_mask(self.nbits).and_(bigval) )

    elif isinstance(w_other, W_SmallBits):
      raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during <<= non-blocking assignment, "
                                      "but here LHS Bits%d > RHS Bits%d.\n"
                                      "- Suggestion: LHS <<= sext/zext(RHS, nbits/Type)", self.nbits, w_other.nbits )
    else:
      w_other_bits = space.call_method(w_other, 'to_bits')
      if isinstance(w_other_bits, W_BigBits):
        if nbits != w_other_bits.nbits:
          if nbits > w_other_bits.nbits:
            raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS to_bits() during <<= non-blocking assignment, "
                                            "but here LHS Bits%d > RHS Bits%d.\n"
                                            "- Suggestion: LHS <<= zext/sext(RHS, nbits/Type)", nbits, w_other_bits.nbits)
          else:
            raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during <<= non-blocking assignment, "
                                            "but here LHS Bits%d < RHS Bits%d.\n"
                                            "- Suggestion: LHS <<= zext/sext(RHS, nbits/Type)", nbits, w_other_bits.nbits)
        return W_BigBitsWithNext( nbits, self.bigval, w_other_bits.bigval )

      elif isinstance(w_other_bits, W_SmallBits):
        raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS to_bits() during <<= non-blocking assignment, "
                                        "but here LHS Bits%d > RHS Bits%d.\n"
                                        "- Suggestion: LHS <<= zext/sext(RHS, nbits/Type)", nbits, w_other_bits.nbits )
      else:
        raise oefmt(space.w_ValueError, "RHS to_bits() must return Bits object!")

  def _descr_flip(self, space):
    raise oefmt(space.w_TypeError, "_flip cannot be called on '%T' objects which has no _next", self)

  #-----------------------------------------------------------------------
  # Miscellaneous methods for string format
  #-----------------------------------------------------------------------

  def descr_bin(self, space):
    w_data = space.newtext( self.bigval.format(BASE2) )
    return space.newtext("0b").descr_add( space, w_data.descr_zfill(space, self.nbits) )

  def descr_oct(self, space):
    w_data = space.newtext( self.bigval.format(BASE8) )
    return space.newtext("0o").descr_add( space, w_data.descr_zfill(space, ((self.nbits-1)/3)+1) )

  def descr_hex(self, space):
    w_data = space.newtext( self.bigval.format(BASE16) )
    return space.newtext("0x").descr_add( space, w_data.descr_zfill(space, ((self.nbits-1)/4)+1) )

  def descr_str(self, space):
    w_data = space.newtext( self.bigval.format(BASE16) )
    return w_data.descr_zfill(space, ((self.nbits-1)/4)+1)

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

  def _descr_copy(self):
    return W_BigBitsWithNext( self.nbits, self.bigval, self.next_bigval )

  def _descr_ilshift(self, space, w_other):
    nbits = self.nbits

    if isinstance(w_other, W_BigBits):
      if nbits != w_other.nbits:
        if nbits > w_other.nbits:
          raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during <<= non-blocking assignment, "
                                          "but here LHS Bits%d > RHS Bits%d.\n"
                                          "- Suggestion: LHS <<= zext/sext(RHS, nbits/Type)", nbits, w_other.nbits)
        else:
          raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during <<= non-blocking assignment, "
                                          "but here LHS Bits%d < RHS Bits%d.\n"
                                          "- Suggestion: LHS <<= trunc(RHS, nbits/Type)", nbits, w_other.nbits)
      self.next_bigval = w_other.bigval

    elif isinstance(w_other, W_IntObject): # int MUST fit Bits64+
      self.next_bigval = get_long_mask(nbits).int_and_(w_other.intval)

    elif isinstance(w_other, W_LongObject):
      bigval = w_other.num
      if _rbigint_check_exceed_nbits( bigval, nbits ):
        raise oefmt(space.w_ValueError, "RHS value %s of <<= is too wide for LHS Bits%d!\n" \
                                        "(Bits%d only accepts %s <= value <= %s)",
                                        bigval.format(BASE16, prefix='0x'), nbits, nbits,
                                        hex(get_int_lower(nbits)), hex(get_int_mask(nbits)))

      self.next_bigval = get_long_mask(nbits).and_(bigval)

    elif isinstance(w_other, W_SmallBits):
      raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during <<= non-blocking assignment, "
                                      "but here LHS Bits%d > RHS Bits%d.\n"
                                      "- Suggestion: LHS <<= zext/sext(RHS, nbits/Type)", nbits, w_other.nbits )
    else:
      w_other_bits = space.call_method(w_other, 'to_bits')
      if isinstance(w_other_bits, W_BigBits):
        if nbits != w_other_bits.nbits:
          if nbits > w_other_bits.nbits:
            raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS to_bits() during <<= non-blocking assignment, "
                                            "but here LHS Bits%d > RHS Bits%d.\n"
                                            "- Suggestion: LHS <<= zext/sext(RHS, nbits/Type)", nbits, w_other_bits.nbits)
          else:
            raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS during <<= non-blocking assignment, "
                                            "but here LHS Bits%d < RHS Bits%d.\n"
                                            "- Suggestion: LHS <<= zext/sext(RHS, nbits/Type)", nbits, w_other_bits.nbits)
        self.next_bigval = w_other_bits.bigval

      elif isinstance(w_other_bits, W_SmallBits):
        raise oefmt(space.w_ValueError, "Bitwidth of LHS must be equal to RHS to_bits() during <<= non-blocking assignment, "
                                        "but here LHS Bits%d > RHS Bits%d.\n"
                                        "- Suggestion: LHS <<= zext/sext(RHS, nbits/Type)", nbits, w_other_bits.nbits )
      else:
        raise oefmt(space.w_ValueError, "RHS to_bits() must return Bits object!")

    return self

  def _descr_flip(self, space):
    self.bigval = self.next_bigval
