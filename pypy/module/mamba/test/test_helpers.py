from hypothesis import strategies, given, example, assume
from rpython.rlib.rbigint     import rbigint, SHIFT, BASE8, BASE16
from pypy.module.mamba.helper_funcs import (setitem_long_long_helper,
        _rbigint_rshift_maskoff, _rbigint_rshift_maskoff_retint)

@strategies.composite
def setitem_inputs(draw):
    nbits = draw(strategies.integers(64, 1024))
    # only big bits for now
    value = rbigint.fromlong(draw(strategies.integers(min_value=0, max_value=(1<<nbits)-1)))
    start = draw(strategies.integers(0, nbits - 1))
    stop = draw(strategies.integers(start + 1, nbits))
    other = rbigint.fromlong(draw(strategies.integers(min_value=0, max_value=(1<<(stop - start)) - 1)))
    return nbits, value, other, start, stop

def bitify(nbits, value):
    # turn into a list of chars 0/1 of length nbits
    bits = list(value.format("01"))
    bits.reverse()
    bits.extend(['0'] * (nbits - len(bits)))
    return bits

def cmpbits(result, bits):
    if type(result) in (int, long):
        assert bin(result)[2:].zfill(len(bits)) == "".join(reversed(bits))
    else:
        assert result.format("01").zfill(len(bits)) == "".join(reversed(bits))

@example((257, rbigint.fromlong(1 << (63 * 2)), rbigint.fromlong(1 << 63), 0, 127))
@given(setitem_inputs())
def test_setitem_long_long_helper(input):
    nbits, value, other, start, stop = input

    # compute result by using lists of bits as oracle
    bits = bitify(nbits, value)
    assert nbits == len(bits)
    otherbits = bitify(stop - start, other)
    assert stop - start == len(otherbits)
    bits[start: stop] = otherbits
    result = setitem_long_long_helper(value, other, start, stop)
    cmpbits(result, bits)

@strategies.composite
def rshift_maskoff_inputs(draw):
    nbits = draw(strategies.integers(64, 1024))
    # only big bits for now
    value = rbigint.fromlong(draw(strategies.integers(min_value=0, max_value=(1<<nbits)-1)))
    shamt = draw(strategies.integers(0, nbits - 1))
    masklen = draw(strategies.integers(1, nbits - shamt))
    return nbits, value, shamt, masklen

@given(rshift_maskoff_inputs())
def test_rshift_maskoff(input):
    nbits, value, shamt, masklen = input
    print nbits, value, shamt, masklen
    bits = bitify(nbits, value)
    resbits = bits[shamt: shamt + masklen]
    res = _rbigint_rshift_maskoff(value, shamt, masklen)
    cmpbits(res, resbits)
    if masklen < SHIFT:
        res = _rbigint_rshift_maskoff_retint(value, shamt, masklen)
        cmpbits(res, resbits)
