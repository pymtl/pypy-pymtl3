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

