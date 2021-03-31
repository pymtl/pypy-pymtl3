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
Then package pypy to be able to get stow-ed.

```
% py2 ../tool/release/package.py --archive-name=pypy-install
```

Go to this tmp folder and copy it to stow packages
```
% cd /tmp/usession-master-<your netid>/build
% cp -rf pypy-install $STOW_PKGS_GLOBAL_PREFIX/pkgs
% cd $STOW_PKGS_GLOBAL_PREFIX/pkgs
```

Now, create a virtual environment:

```
% cd $VENV_PKGS_GLOBAL_PREFIX
% $STOW_PKGS_GLOBAL_PREFIX/pkgs/pypy-install/bin/pypy -m venv pypy-pymtl
```
