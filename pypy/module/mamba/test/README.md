# How to run test on Mamba module
 - You need to have pytest version < 3.0.0. I have a virtual environment installed at `/work/global/lc873/work/pymtl/venv/pytest`
 - cd into mamba test dir `pypy/module/mamba/test/`
 - Call `./../../../pytest.py ./test_bits.py -vs`
 - Note that it takes a long time to start since this runs on top of Python interpreter
