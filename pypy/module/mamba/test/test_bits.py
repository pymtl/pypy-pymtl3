import os
import pytest

class AppTestBits:
    spaceconfig = dict(usemodules=['mamba'])

    def test_import_mamba(self):
        import mamba, sys
        assert 0 == 0

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


