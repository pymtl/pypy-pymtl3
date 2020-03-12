import os
import pytest

class AppTestBits:
    spaceconfig = dict(usemodules=['mamba'])

    def test_import_mamba(self):
        import mamba, sys
        assert 0 == 0

    def test_bits_getitem(self):
        import mamba
        b = mamba.Bits(8, 0b10110010)
        assert b[0] == mamba.Bits(1, 0)
        assert b[1] == mamba.Bits(1, 1)
        assert b[2] == mamba.Bits(1, 0)
        assert b[3] == mamba.Bits(1, 0)
        assert b[4] == mamba.Bits(1, 1)
        assert b[5] == mamba.Bits(1, 1)
        assert b[6] == mamba.Bits(1, 0)
        assert b[7] == mamba.Bits(1, 1)

    def test_bits_setitem(self):
        import mamba
        b = mamba.Bits(8, 0b10110010)
        b[0] = 1
        assert b == 0b10110011
        with raises(ValueError):
            b[0] = 12

    def test_bits_rsub(self):
        import mamba
        b = mamba.Bits(10, 18)
        assert 20 - b == 2
        assert -1 - b == 1024 - 1 - 18

        b = mamba.Bits(100, 18)
        assert 20 - b == 2
        assert -1 - b == (2**100) - 1 - 18

    def test_bits_shift(self):
        import mamba
        b = mamba.Bits(10, 1)
        assert b << 10 == 0
        assert b << 1 == 2
        assert b << 4 == 1 << 4
        assert b << mamba.Bits(10, 100) == 0
        assert b << mamba.Bits(10, 1) == 2
        assert b << mamba.Bits(10, 4) == 1 << 4

        b = mamba.Bits(10, 0b10000)
        assert b >> 4 == 1
        assert b >> 10 == 0
        assert b >> 2 == 0b100
        assert b >> mamba.Bits(10, 4) == 1
        assert b >> mamba.Bits(10, 10) == 0
        assert b >> mamba.Bits(10, 2) == 0b100

        b = mamba.Bits(100, 1)
        assert b << 100 == 0
        assert b << 1 == 2
        assert b << 4 == 1 << 4
        assert b << mamba.Bits(10, 100) == 0
        assert b << mamba.Bits(10, 1) == 2
        assert b << mamba.Bits(10, 4) == 1 << 4

        b = mamba.Bits(100, 0b10000)
        assert b >> 4 == 1
        assert b >> 10 == 0
        assert b >> 2 == 0b100
        assert b >> mamba.Bits(10, 4) == 1
        assert b >> mamba.Bits(10, 10) == 0
        assert b >> mamba.Bits(10, 2) == 0b100

    def test_mixed_cmp(self):
        import mamba
        s = mamba.Bits(10, 1)
        b = mamba.Bits(100, 1)
        assert s == s
        assert s == b
        assert b == s
        assert b == b

    def test_mixed_arithmetic(self):
        import mamba
        s = mamba.Bits(10, 1)
        b = mamba.Bits(100, 1)
        assert s + s == 2
        assert b + s == 2
        assert s + b == 2
        assert b + b == 2
        assert s & s == 1
        assert b & s == 1
        assert s & b == 1
        assert b & b == 1

    def test_bits_str(self):
        import mamba, sys
        b = mamba.Bits(8,42)
        assert str(b) == '2a'
        b = mamba.Bits(32,42)
        assert str(b) == '0000002a'
        b = mamba.Bits(32,48879)
        assert str(b) == '0000beef'
        b = mamba.Bits(512,13907095861846720239)
        assert str(b) == '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c0ffee00deadbeef'

    def test_bits_repr(self):
        import mamba, sys
        b = mamba.Bits(8,42)
        assert repr(b) == 'Bits8( 0x2a )'
        b = mamba.Bits(32,42)
        assert repr(b) == 'Bits32( 0x0000002a )'
        b = mamba.Bits(32,48879)
        assert repr(b) == 'Bits32( 0x0000beef )'
        b = mamba.Bits(512,13907095861846720239)
        assert repr(b) == 'Bits512( 0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c0ffee00deadbeef )'

    def test_ilshift_create_bits_with_next(self):
        import mamba, sys
        x = b = mamba.Bits(8,42)
        n = mamba.Bits(8,43)
        b <<= n
        assert x is not b
        assert x == b

    def test_flip_on_bits(self):
        import mamba, sys
        b = mamba.Bits(8,42)
        raises(TypeError, b._flip)

    def test_flip_on_bits_with_next(self):
        import mamba, sys
        x = b = mamba.Bits(8,42)
        n = mamba.Bits(8,43)
        b <<= n
        assert x is not b
        assert x == b
        b._flip()
        assert n == b
        assert x != b

    def test_ilshift_on_bits_with_next(self):
        import mamba, sys
        b8 = mamba.Bits(8,42)
        n1 = mamba.Bits(8,43)
        b8 <<= n1
        # now b8 should have next_intval field
        b8._flip()
        assert b8 == n1
        # do the assignment again
        n2 = mamba.Bits(8,44)
        tmp = b8
        b8 <<= n2
        assert b8 is tmp
        b8._flip()
        assert b8 == n2

    def test_ilshift_mod_after_buffering(self):
        import mamba, sys
        b8 = mamba.Bits(8,42)
        n  = mamba.Bits(8,43)
        b8 <<= n
        n = n + 23
        assert n == mamba.Bits(8,66)
        b8._flip()
        assert b8 == mamba.Bits(8,43)

    def test_ilshift_buffer_preserve(self):
        import mamba, sys
        b8 = mamba.Bits(8,42)
        n  = mamba.Bits(8,43)
        b8 <<= n
        b8._flip()
        assert b8 == mamba.Bits(8,43)
        b8[0:1] = 0
        assert b8 == mamba.Bits(8,42)
        b8._flip()
        assert b8 == mamba.Bits(8,43)

    def test_ilshift_create_bits_with_next_big(self):
        import mamba, sys
        x = b = mamba.Bits(256,42)
        n = mamba.Bits(256,43)
        b <<= n
        assert x is not b
        assert x == b

    def test_flip_on_bits_big(self):
        import mamba, sys
        b = mamba.Bits(256,42)
        raises(TypeError, b._flip)

    def test_flip_on_bits_with_next_big(self):
        import mamba, sys
        x = b = mamba.Bits(256,42)
        n = mamba.Bits(256,43)
        b <<= n
        assert x is not b
        assert x == b
        b._flip()
        assert n == b
        assert x != b

    def test_ilshift_on_bits_with_next_big(self):
        import mamba, sys
        b256 = mamba.Bits(256,42)
        n1 = mamba.Bits(256,43)
        b256 <<= n1
        # now b256 should have next_intval field
        b256._flip()
        assert b256 == n1
        # do the assignment again
        n2 = mamba.Bits(256,44)
        tmp = b256
        b256 <<= n2
        assert b256 is tmp
        b256._flip()
        assert b256 == n2

    def test_ilshift_mod_after_buffering_big(self):
        import mamba, sys
        b256 = mamba.Bits(256,42)
        n  = mamba.Bits(256,43)
        b256 <<= n
        n = n + 23
        assert n == mamba.Bits(256,66)
        b256._flip()
        assert b256 == mamba.Bits(256,43)

    def test_ilshift_buffer_preserve_big(self):
        import mamba, sys
        b256 = mamba.Bits(256,42)
        n  = mamba.Bits(256,43)
        b256 <<= n
        b256._flip()
        assert b256 == mamba.Bits(256,43)
        b256[0:1] = 0
        assert b256 == mamba.Bits(256,42)
        b256._flip()
        assert b256 == mamba.Bits(256,43)
