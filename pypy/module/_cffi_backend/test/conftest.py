import os
import pytest
import sys

if sys.platform == 'win32':
    def pytest_configure(config):
        if config.getoption('direct_apptest'):
            return
        # Set up the compiler via rpython.platform when run with CPython 2.7
        from rpython.translator.platform import host
        for key in ('PATH', 'LIB', 'INCLUDE'):
            os.environ[key] = host.c_environ[key]
        os.environ['DISTUTILS_USE_SDK'] = "1"
        os.environ['MSSdk'] = "1"
