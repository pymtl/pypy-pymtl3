from hypothesis import strategies, given, example, assume
from rpython.rlib.rbigint     import rbigint, SHIFT, BASE8, BASE16
from pypy.module.mamba.helper_funcs import setitem_long_long_helper

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
    assert result.format("01").zfill(len(bits)) == "".join(reversed(bits))


