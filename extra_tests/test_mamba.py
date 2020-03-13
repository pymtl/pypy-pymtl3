# to run, first install the dependencies on the compiled pypy:
# $ pypy -m ensurepip
# $ pypy -m pip install hypothesis pytest
# $ pypy -m pytest test_mamba.py

import sys
import operator

from mamba import Bits
from hypothesis import strategies, given, example, assume

@strategies.composite
def bits(draw):
  if draw(strategies.booleans()):
    # small bits
    nbits = draw(strategies.integers(min_value=1, max_value=63))
  else:
    # big bits
    nbits = draw(strategies.integers(min_value=64, max_value=512))
  value = draw(strategies.integers(min_value=-(1<<(nbits-1)), max_value=(1<<nbits)-1))
  return Bits(nbits, value)

def several_bits(n, same_bitwidth=True):
    @strategies.composite
    def several_bits(draw):
      if same_bitwidth or draw(strategies.booleans()):
        # all same bitwidth
        first = draw(bits())
        return [first] + [Bits(first.nbits, draw(strategies.integers(min_value=-(1<<(first.nbits-1)), max_value=(1<<first.nbits)-1)))
                  for i in range(n - 1)]
      return [draw(bits()) for i in range(n)]
    return several_bits()


def as_long(val):
  if isinstance(val, Bits):
    val = int(val)
  return val + 2 ** 100 - 2 ** 100

@given(several_bits(2), strategies.sampled_from([operator.add, operator.mul, operator.or_, operator.and_, operator.xor]))
def test_arith_commutative(bits, op):
  bits1, bits2 = bits
  res = op(bits1, bits2)
  assert res == op(bits2, bits1)  # commutativity
  assert res == Bits(max(bits1.nbits, bits2.nbits), op(int(bits1), int(bits2)), trunc=True)
  assert op(bits1, int(bits2)) == res
  assert op(bits1, as_long(bits2)) == res
  assert op(int(bits2), bits1) == res
  assert op(as_long(bits2), bits1) == res

  assert op(int(bits1), bits2) == res
  assert op(as_long(bits1), bits2) == res
  assert op(bits2, int(bits1)) == res
  assert op(bits2, as_long(bits1)) == res

@given(several_bits(2))
def test_arith_sub(bits):
  bits1, bits2 = bits
  op = operator.sub
  # not commutative
  res = op(bits1, bits2)
  assert res == Bits(max(bits1.nbits, bits2.nbits), op(int(bits1), int(bits2)))
  assert op(bits1, int(bits2)) == res
  assert op(bits1, as_long(bits2)) == res

  assert op(int(bits1), bits2) == res
  assert op(as_long(bits1), bits2) == res

@given(several_bits(3), strategies.sampled_from([operator.add, operator.mul, operator.or_, operator.and_, operator.xor]))
def test_associativity(bits, op):
  bits1, bits2, bits3 = bits
  res1 = op(bits1, op(bits2, bits3))
  res2 = op(op(bits1, bits2), bits3)
  assert res1 == res2

@given(bits())
def test_getitem(bits):
  binary = bin(bits)[2:].rjust(bits.nbits, "0")[::-1]
  assert len(binary) == bits.nbits
  for index in range(0, bits.nbits):
    assert bits[index] == int(binary[index], 2)

@given(several_bits(2, False), strategies.sampled_from(
        [(operator.eq, operator.eq),
         (operator.lt, operator.gt),
         (operator.le, operator.ge),
         (operator.ne, operator.ne)])
)
def test_cmp(bits, comparators):
  bits1, bits2 = bits
  op, inv_op = comparators
  res = op(bits1, bits2)
  print(bits1, bits2, res)
  assert res == inv_op(bits2, bits1)

  assert res == op(int(bits1), int(bits2))

