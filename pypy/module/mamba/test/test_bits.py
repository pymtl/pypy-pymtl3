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
        assert b[mamba.Bits(1, 1)] == mamba.Bits(1, 1)
        assert b[mamba.Bits(100, 1)] == mamba.Bits(1, 1)
        assert b[2] == mamba.Bits(1, 0)
        assert b[3] == mamba.Bits(1, 0)
        assert b[4] == mamba.Bits(1, 1)
        assert b[5] == mamba.Bits(1, 1)
        assert b[6] == mamba.Bits(1, 0)
        assert b[7] == mamba.Bits(1, 1)

        assert b[0:2] == mamba.Bits(2, 0b10)
        assert b[mamba.Bits(1, 0):2] == mamba.Bits(2, 0b10)
        assert b[0:mamba.Bits(3, 2)] == mamba.Bits(2, 0b10)
        assert b[mamba.Bits(100, 0):2] == mamba.Bits(2, 0b10)
        assert b[0:mamba.Bits(300, 2)] == mamba.Bits(2, 0b10)

        with raises(IndexError):
            b[0:3:2]

    def test_bigbits_getitem(self):
        import mamba
        b = mamba.Bits(80, 0b10110010)
        assert b[0] == mamba.Bits(1, 0)
        assert b[1] == mamba.Bits(1, 1)
        assert b[mamba.Bits(1, 1)] == mamba.Bits(1, 1)
        assert b[mamba.Bits(100, 1)] == mamba.Bits(1, 1)
        assert b[2] == mamba.Bits(1, 0)
        assert b[3] == mamba.Bits(1, 0)
        assert b[4] == mamba.Bits(1, 1)
        assert b[5] == mamba.Bits(1, 1)
        assert b[6] == mamba.Bits(1, 0)
        assert b[7] == mamba.Bits(1, 1)

        assert b[0:2] == mamba.Bits(2, 0b10)
        assert b[mamba.Bits(1, 0):2] == mamba.Bits(2, 0b10)
        assert b[0:mamba.Bits(3, 2)] == mamba.Bits(2, 0b10)
        assert b[mamba.Bits(100, 0):2] == mamba.Bits(2, 0b10)
        assert b[0:mamba.Bits(300, 2)] == mamba.Bits(2, 0b10)

        initval = 0b10110010 << 70 | 0b110101011
        b = mamba.Bits(80, initval)
        assert b[0:70] == mamba.Bits(70, 0b110101011)
        assert b[5:75] == mamba.Bits(70, (initval >> 5) & (2 ** 70 - 1))

        with raises(IndexError):
            b[0:3:2]

    def test_bits_getitem_bug(self):
        import mamba
        b = mamba.Bits(100, 1)[63] # used to crash
        assert b == 0

    def test_smallbits_setitem(self):
        import mamba
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        b = mamba.Bits(8, 0b10110010)
        b[0] = 1
        assert b == 0b10110011
        b = mamba.Bits(8, 0b10110010)
        b[0] = make_long(1)
        assert b == 0b10110011
        b = mamba.Bits(8, 0b10110010)
        b[0] = mamba.Bits(1, 1)
        assert b == 0b10110011
        with raises(ValueError):
            b[0] = mamba.Bits(80, 1)

    def test_bigbits_setitem(self):
        import mamba
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        b = mamba.Bits(80, 0b10110010)
        b[0] = 1
        assert b == 0b10110011
        b = mamba.Bits(80, 0b10110010)
        b[0] = mamba.Bits(1, 1)
        assert b == 0b10110011
        b = mamba.Bits(80, 0b10110010)
        b[0] = make_long(1)
        assert b == 0b10110011

        with raises(ValueError):
            b[0] = 12

    def test_smallbits_setslice(self):
        import mamba
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        b = mamba.Bits(8, 0b10110001)
        b[0:2] = 0b10
        assert b == 0b10110010

        b = mamba.Bits(8, 0b10110010)
        b[0:2] = 0b1
        assert b == 0b10110001

        b = mamba.Bits(8, 0b10110001)
        b[0:2] = make_long(0b10)
        assert b == 0b10110010

        b = mamba.Bits(8, 0b10110001)
        b[0:2] = mamba.Bits(2, 0b10)
        assert b == 0b10110010

        with raises(ValueError):
            b[0:2] = mamba.Bits(10, 0)

    def test_bigbits_setslice(self):
        import mamba
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        b = mamba.Bits(80, 0b10110001)
        b[0:2] = 0b10
        assert b == 0b10110010

        b = mamba.Bits(80, 0b10110010)
        b[0:2] = 0b1
        assert b == 0b10110001

        b = mamba.Bits(80, 0b10110001)
        b[0:2] = make_long(0b10)
        assert b == 0b10110010

        b = mamba.Bits(80, 0b10110001)
        b[0:3] = -2
        assert b == 0b10110110

        b = mamba.Bits(80, 0b10110001)
        b[0:3] = make_long(-2)
        assert b == 0b10110110

        b = mamba.Bits(80, 0b10110001)
        b[0:2] = mamba.Bits(2, 0b10)
        assert b == 0b10110010

        with raises(ValueError):
            b[0:2] = mamba.Bits(10, 0)

        b = mamba.Bits(150, 0)
        b[0:80] = mamba.Bits(80, 4)
        assert b == 4

        b = mamba.Bits(150, 0)
        b[0:80] = mamba.Bits(80, 4 << 60)
        assert b == 4 << 60

        with raises(ValueError):
            b[0:80] = mamba.Bits(100, 0)

    def test_setitem_crash(self):
        from mamba import Bits
        input = Bits(465, 0x00095700000000000000003f950000000000000000000000000000000000000000000000000000000000000000000000000000000000000000f5d )
        input[363: 441] = Bits(78, 0x000000000000000000b0)

    def test_setitem_crash(self):
        from mamba import Bits
        input = Bits(465, 0x00095700000000000000003f950000000000000000000000000000000000000000000000000000000000000000000000000000000000000000f5d )
        input[363: 441] = Bits(78, 0x000000000000000000b0)

    def test_bits_new(self):
        import mamba
        with raises(ValueError):
            mamba.Bits(0, 1)
        with raises(ValueError):
            mamba.Bits(-1, 1)
        with raises(ValueError):
            mamba.Bits(4, mamba.Bits(3, 2))

    def test_nbits(self):
        import mamba
        for i in range(1, 100):
            assert mamba.Bits(i, 0).nbits == i

    def test_sub_bug(self):
        from mamba import Bits
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        for bitwidth in [2, 8, 32, 64, 100]:
            res = Bits(bitwidth, 0) - Bits(bitwidth, 1)
            assert res == Bits(bitwidth, 2 ** bitwidth - 1)
            assert Bits(bitwidth, 0) - 1 == res
            assert Bits(bitwidth, 0) - make_long(1) == res

    def test_bits_rsub(self):
        import mamba
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        b = mamba.Bits(10, 18)
        assert 20 - b == 2
        assert make_long(20) - b == 2
        assert make_long(1) - b == 1024 + 1 - 18
        with raises(ValueError):
          assert -1 - b == 1024 - 1 - 18 # rsub also only allow positive int

        b = mamba.Bits(100, 18)
        assert 20 - b == 2
        assert 1 - b == (2**100) + 1 - 18
        assert make_long(20) - b == 2
        assert make_long(1) - b == (2**100) + 1 - 18

    def test_smallbits_shift(self):
        import mamba
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        b = mamba.Bits(10, 1)
        assert b << 10 == 0
        assert b << 1 == 2
        assert b << 4 == 1 << 4
        assert b << make_long(10) == 0
        assert b << make_long(1) == 2
        assert b << make_long(4) == 1 << 4
        assert b << mamba.Bits(10, 100) == 0
        assert b << mamba.Bits(10, 1) == 2
        assert b << mamba.Bits(10, 4) == 1 << 4

        with raises(ValueError):
          b << 1024
        with raises(ValueError):
          b << -1
        with raises(ValueError):
          assert b << mamba.Bits(100, 100) == 0
        with raises(ValueError):
          assert b << mamba.Bits(100, 1) == 2
        with raises(ValueError):
          assert b << mamba.Bits(100, 4) == 1 << 4

        b = mamba.Bits(10, 0b10000)
        assert b >> 4 == 1
        assert b >> 10 == 0
        assert b >> 2 == 0b100
        assert b >> make_long(4) == 1
        assert b >> make_long(10) == 0
        assert b >> make_long(2) == 0b100
        assert b >> mamba.Bits(10, 4) == 1
        assert b >> mamba.Bits(10, 10) == 0
        assert b >> mamba.Bits(10, 2) == 0b100
        with raises(ValueError):
          b >> 1024
        with raises(ValueError):
          b >> -1
        with raises(ValueError):
          assert b >> mamba.Bits(100, 4) == 1
        with raises(ValueError):
          assert b >> mamba.Bits(100, 10) == 0
        with raises(ValueError):
          assert b >> mamba.Bits(100, 2) == 0b100

    def test_bigbits_shift(self):
        import mamba
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        b = mamba.Bits(100, 1)
        assert b << 100 == 0
        assert b << 1 == 2
        assert b << 4 == 1 << 4
        assert b << make_long(100) == 0
        assert b << make_long(1) == 2
        assert b << make_long(4) == 1 << 4

        with raises(ValueError):
          b << (2**100)
        with raises(ValueError):
          b << -1
        with raises(ValueError):
          assert b << mamba.Bits(10, 100) == 0
        with raises(ValueError):
          assert b << mamba.Bits(10, 1) == 2
        with raises(ValueError):
          assert b << mamba.Bits(10, 4) == 1 << 4
        assert b << mamba.Bits(100, 100) == 0
        assert b << mamba.Bits(100, 1) == 2
        assert b << mamba.Bits(100, 4) == 1 << 4

        b = mamba.Bits(100, 0b10000)
        assert b >> 4 == 1
        assert b >> 10 == 0
        assert b >> 2 == 0b100
        assert b >> make_long(4) == 1
        assert b >> make_long(10) == 0
        assert b >> make_long(2) == 0b100

        with raises(ValueError):
          b >> (2**100)
        with raises(ValueError):
          b >> -1
        with raises(ValueError):
          assert b >> mamba.Bits(10, 4) == 1
        with raises(ValueError):
          assert b >> mamba.Bits(10, 10) == 0
        with raises(ValueError):
          assert b >> mamba.Bits(10, 2) == 0b100
        assert b >> mamba.Bits(100, 4) == 1
        assert b >> mamba.Bits(100, 10) == 0
        assert b >> mamba.Bits(100, 2) == 0b100
        with raises(ValueError):
          assert b >> (1 << 2000) == 0

    def test_invert(self):
        import mamba
        b = mamba.Bits(8, 17)
        assert ~b == mamba.Bits(8, ~17)
        b = mamba.Bits(80, 17)
        assert ~b == mamba.Bits(80, ~17)

    def test_mixed_cmp(self):
        import mamba
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        l1 = [1, make_long(1), mamba.Bits(100, 1)]
        l2 = [5, make_long(5), mamba.Bits(100, 5)]
        for a in l1:
            for b in l1:
                assert a == b
            for b in l2:
                assert a != b
                assert b != a
        l1 = [1, make_long(1), mamba.Bits(10, 1)]
        l2 = [5, make_long(5), mamba.Bits(10, 5)]
        for a in l1:
            for b in l1:
                assert a == b
            for b in l2:
                assert a != b
                assert b != a
        with raises(ValueError):
            mamba.Bits(10,1) == mamba.Bits(11,1)
        with raises(ValueError):
            mamba.Bits(10,1) == mamba.Bits(111,1)
        with raises(ValueError):
            mamba.Bits(110,1) == mamba.Bits(111,1)

    def test_mixed_arithmetic(self):
        import mamba
        def make_long(x): return x + 2 ** 100 - 2 ** 100
        l = [mamba.Bits(100, 1), 1, make_long(1)]
        for a in l:
            for b in l:
                assert a + b == b + a == 2
                assert a & b == b & a == 1
        l = [mamba.Bits(10, 1), 1, make_long(1)]
        for a in l:
            for b in l:
                assert a + b == b + a == 2
                assert a & b == b & a == 1
        assert mamba.Bits(64,1) + int(mamba.Bits(64, 0xffffffffffffffff )) == 0

        with raises(ValueError):
            mamba.Bits(10,1) + mamba.Bits(11,1)
        with raises(ValueError):
            mamba.Bits(10,1) + mamba.Bits(111,1)
        with raises(ValueError):
            mamba.Bits(110,1) + mamba.Bits(111,1)
        with raises(ValueError):
            mamba.Bits(10,1) & mamba.Bits(11,1)
        with raises(ValueError):
            mamba.Bits(10,1) & mamba.Bits(111,1)
        with raises(ValueError):
            mamba.Bits(110,1) & mamba.Bits(111,1)

    def test_add_ovf_bug(self):
        import mamba
        b = mamba.Bits(1, 1)
        assert b + b == 0

    def test_floordiv(self):
        import mamba
        x = mamba.Bits( 4, 5 )
        y = mamba.Bits( 4, 4 )
        assert x // y == 1
        assert x // mamba.Bits(4, 4) == 1
        assert x // 4 == 1
        y = mamba.Bits( 4, 6 )
        assert x // y == 0
        assert x // 6 == 0
        assert 6 // x == 1
        with raises( ValueError ):
            x // -1
        with raises( ValueError ):
            x // 100000000000000000000000000

        with raises( ValueError ):
          a = mamba.Bits(4,3) // mamba.Bits(3,1)

    def test_rfloordiv(self):
        import mamba
        x = mamba.Bits(4, 3)
        y = 8
        z = y // x
        assert z.nbits == 4 and z.uint() == 2
        y = 16
        with raises( ValueError ):
            z = y // x

        x = mamba.Bits(400, 3**100)
        y = 2**300
        z = y // x
        assert z.nbits == 400 and z.uint() == 3952512273801159033450488434333219932310210
        y = 2**400
        with raises( ValueError ):
            z = y // x

    def test_mod(self):
        import mamba
        x = mamba.Bits( 4, 5 )
        y = mamba.Bits( 4, 4 )
        assert x % y == 1
        assert x % mamba.Bits(4, 4) == 1
        assert x % 4 == 1
        y = mamba.Bits( 4, 6 )
        assert x % y == 5
        assert x % 6 == 5
        assert 6 % x == 1
        with raises( ValueError ):
            x % -1
        with raises( ValueError ):
            x % 100000000000000000000000000

        with raises( ValueError ):
          a = mamba.Bits(4,3) % mamba.Bits(3,1)

    def test_rmod(self):
        import mamba
        x = mamba.Bits(4, 3)
        y = 8
        z = y % x
        assert z.nbits == 4 and z.uint() == 2
        y = 16
        with raises( ValueError ):
            z = y % x

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
        assert repr(b) == 'Bits8(0x2a)'
        b = mamba.Bits(32,42)
        assert repr(b) == 'Bits32(0x0000002a)'
        b = mamba.Bits(32,48879)
        assert repr(b) == 'Bits32(0x0000beef)'
        b = mamba.Bits(512,13907095861846720239)
        assert repr(b) == 'Bits512(0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c0ffee00deadbeef)'

    def test_bits_bin_oct_hex(self):
        import mamba, sys
        assert mamba.Bits(15,35).bin() == '0b000000000100011'
        assert mamba.Bits(15,35).oct() == '0o00043'
        assert mamba.Bits(15,35).hex() == '0x0023'

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

    def test_ilshift_other_class(self):
        import mamba, sys

        class B_4_8(object):
          def __init__( self, a, b ):
            self.a = mamba.Bits(4,a)
            self.b = mamba.Bits(8,b)
          def to_bits( self ):
            ret = mamba.Bits(12, 0)
            ret[0:4]  = self.a
            ret[4:12] = self.b
            return ret

        class B_88_44(object):
          def __init__( self, a, b ):
            self.a = mamba.Bits(88,a)
            self.b = mamba.Bits(44,b)
          def to_bits( self ):
            ret = mamba.Bits(132, 0)
            ret[0:88]  = self.a
            ret[88:132] = self.b
            return ret

        b = B_4_8(4,2)
        a = mamba.Bits(12, 0)
        a <<= b
        assert a == 0
        a._flip()
        assert a == 36

        c = B_88_44(4,2)
        with raises(ValueError):
          a <<= c

        a = mamba.Bits(13, 0)
        with raises( ValueError ):
          a <<= b

        a = mamba.Bits(11, 0)
        with raises( ValueError ):
          a <<= b

        d = mamba.Bits(132, 0)
        d <<= c
        assert d == 0
        d._flip()
        assert d == 0x20000000000000000000004

    def test_imatmul_other_class(self):
        import mamba, sys

        class B_4_8(object):
          def __init__( self, a, b ):
            self.a = mamba.Bits(4,a)
            self.b = mamba.Bits(8,b)
          def to_bits( self ):
            ret = mamba.Bits(12, 0)
            ret[0:4]  = self.a
            ret[4:12] = self.b
            return ret

        class B_88_44(object):
          def __init__( self, a, b ):
            self.a = mamba.Bits(88,a)
            self.b = mamba.Bits(44,b)
          def to_bits( self ):
            ret = mamba.Bits(132, 0)
            ret[0:88]  = self.a
            ret[88:132] = self.b
            return ret

        b = B_4_8(4,2)
        a = mamba.Bits(12, 0)
        a.__imatmul__( b )
        assert a == 36

        c = B_88_44(4,2)
        with raises(ValueError):
          a.__imatmul__( c )

        a = mamba.Bits(13, 0)
        with raises( ValueError ):
          a.__imatmul__( b )

        a = mamba.Bits(11, 0)
        with raises( ValueError ):
          a.__imatmul__( b )

        d = mamba.Bits(132, 0)
        d.__imatmul__( c )
        assert d == 0x20000000000000000000004

    def test_concat(self):
        from mamba import Bits, concat
        assert concat(Bits(2, 1), Bits(2, 0b10)) == Bits(4, 0b0110)
        assert concat(Bits(1, 1), Bits(64, 0)) == Bits(65, 1 << 64)
        assert concat(Bits(1, 1), Bits(128, 1<<64)) == Bits(129, (1 << 128) | (1 << 64))
