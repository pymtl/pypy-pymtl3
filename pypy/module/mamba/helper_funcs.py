#=========================================================================
# helper_funcs.py
#=========================================================================
# This file includes helper functions which are really rpython arithmetic
# functions and simplified implementations customized and tuned for Bits
# arithmetics.
# For example, using setitem_long_long/int_helper for Bits __setitem__
# is much faster than doing rbigint lshift, rshift, and AND.
#
# Author : Shunning Jiang
# Date   : March 10, 2020

from rpython.rlib import jit
from rpython.rlib.rbigint import rbigint, SHIFT, NULLDIGIT, ONERBIGINT, \
                                 NULLRBIGINT, _store_digit, _x_int_sub, \
                                 _widen_digit, BASE16

BASE2 = '01'

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

def get_long_lower( i ):
  return LONG_MASKS[ i-1 ].int_add(1).neg()
get_long_lower._always_inline_ = True

def get_int_mask( i ):
  return int((1<<i)-1)
get_int_mask._always_inline_ = True

def get_int_lower( i ):
  return -int(1<<(i-1))
get_int_mask._always_inline_ = True

#-------------------------------------------------------------------------
# Shunning: The following functions are specialized implementations for
# Bits arithmetics. Basically we squash arithmetic ops and ANDing mask to
# the same function to avoid copying and also reduce the constant factor
# based on the return type (int/rbigint).
#-------------------------------------------------------------------------

# This function check if a bigint is within [-2**(nbits-1), 2**nbits)
# It is mostly used to check __new__, @=, <<= 's RHS bitwidth
@jit.elidable
def _rbigint_check_exceed_nbits( v, nbits ):
  if v.sign == 0: return False

  if v.sign > 0:
    # We basically need to emulate 1 << nbits which has (nbits+1) bits
    # Then check if the positive v is larger than or equal to the most
    # significant word
    # e.g. nbits = 62 -> 1 word, 1<<62
    #      nbits = 63 -> 2 word, 1<<0
    #      nbits = 64 -> 2 word, 1<<1
    numwords = nbits / SHIFT # intentionally leave + 1 to later
    msw      = 1 << (nbits - numwords*SHIFT)

    numwords += 1
    vsize    = v.numdigits()
    if vsize < numwords:
      return False
    if vsize > numwords:
      return True
    return v.digit(vsize-1) >= msw # to check >= 2**nbits

  # v.sign < 0:
  # We basically need to emulate 1 << (nbits-1)
  nbits -= 1
  numwords = nbits / SHIFT
  msw      = 1 << (nbits - numwords*SHIFT)
  vsize    = v.numdigits()
  if vsize < numwords:
    return False
  if vsize > numwords:
    return True
  return v.digit(vsize-1) > msw # this is >, to check < -2**(nbits-1)

# This function check if a bigint is within [0, 2**nbits)
# It is mostly used to check the integer operands in binary operations
@jit.elidable
def _rbigint_invalid_binop_operand( v, nbits ):
  if v.sign < 0:  return True
  if v.sign == 0: return False
  # v.sign > 0
  # We basically need to emulate 1 << nbits which has (nbits+1) bits
  # Then check if the positive v is larger than or equal to the most
  # significant word
  # e.g. nbits = 62 -> 1 word, 1<<62
  #      nbits = 63 -> 2 word, 1<<0
  #      nbits = 64 -> 2 word, 1<<1
  numwords = nbits / SHIFT # intentionally leave + 1 to later
  msw      = 1 << (nbits - numwords*SHIFT)

  numwords += 1
  vsize    = v.numdigits()
  if vsize < numwords:
    return False
  if vsize > numwords:
    return True
  return v.digit(vsize-1) >= msw # to check >= 2**nbits

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

  loshift = shamt - wordshift*SHIFT
  hishift = SHIFT - loshift
  ret = value.digit(wordshift) >> loshift

  wordshift += 1
  if wordshift < oldsize:
    ret |= value.digit(wordshift) << hishift

  return ret & get_int_mask(masklen)

_rbigint_rshift_maskoff_retint._always_inline_ = True

# This function implements getidx functionality.
@jit.elidable
def _rbigint_getidx( value, index ):
  wordpos = index / SHIFT
  if wordpos >= value.numdigits(): return 0
  bitpos  = index - wordpos*SHIFT
  return (value.digit(wordpos) >> bitpos) & 1
_rbigint_getidx._always_inline_ = True

# This function implements setidx functionality.
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

# setitem helpers that returns a new rbigint with new slice

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
