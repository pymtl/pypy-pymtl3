
from rpython.rlib.rbigint import rbigint, SHIFT

from pypy.interpreter import gateway
from pypy.interpreter.error import oefmt

from bits import W_Bits, setitem_long_int_helper, setitem_long_long_helper, get_int_mask

# @jit.look_inside_iff(lambda space, args_w:
        # jit.loop_unrolling_heuristic(args_w, len(args_w), 3))
    # case of multiple arguments (at least two).  We unroll it if there
    # are 2 or 3 arguments.
def concat_impl(space, args):
    # import pdb;pdb.set_trace()
    args_w = args.arguments_w
    num_args = len(args_w)
    if num_args == 1:
      return args_w[0]

    nbits = 0
    for i in range(0, len(args_w)):
        arg_w = args_w[i]
        if isinstance( arg_w, W_Bits ):
            nbits += arg_w.nbits
        else:
            raise oefmt(space.w_TypeError,
                        "%d-th argument is of wrong type. Concat only takes Bits objects.",
                        i)

    stop = nbits

    if nbits <= SHIFT: # arg_w.nbits must <= SHIFT

        ret = W_Bits( nbits, 0 )

        for i in range(0, len(args_w)):
            arg_w = args_w[i]
            assert isinstance( arg_w, W_Bits )

            slice_nbits = arg_w.nbits
            start = stop - slice_nbits

            valuemask  = ~(get_int_mask(slice_nbits) << start)
            ret.intval = (ret.intval & valuemask) | (arg_w.intval << start)

            stop = start

    else:
        # ret > SHIFT-bits, need to have rbigint
        ret = W_Bits( nbits, 0, rbigint.fromint(0) )

        for i in range(0, len(args_w)):
            arg_w = args_w[i]
            assert isinstance( arg_w, W_Bits )

            slice_nbits = arg_w.nbits
            start = stop - slice_nbits

            if slice_nbits <= SHIFT:
                ret.bigval = setitem_long_int_helper( ret.bigval, arg_w.intval, start, stop )
            else:
                ret.bigval = setitem_long_long_helper( ret.bigval, arg_w.bigval, start, stop )

            stop = start

    return ret

def concat(space, __args__):
    """concat( v1, v2, v3, ... )"""
    return concat_impl( space, __args__ )
