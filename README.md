BRG README for this customized PyPy
===================================

Installation
------------
Get a Python2.7 or PyPy2.7 (translates RPython faster) first.
https://www.pypy.org/download.html Get the appropriate binary depending on the linux distribution. Unzip it

```
% git clone git@github.com:pymtl/pypy-pymtl3.git
% cd pypy-pymtl3/pypy/goal
% alias py2=<pypy2.7 you just downloaded or cpython2.7> 
% py2 ../../rpython/bin/rpython --translation-jit_opencoder_model big -Ojit targetpypystandalone
```
