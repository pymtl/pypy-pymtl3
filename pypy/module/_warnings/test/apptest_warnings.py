# spaceconfig = {"usemodules": ["_warnings"]}

import warnings
import _warnings

import io
import sys
import __pypy__

def test_defaults():
    assert _warnings._onceregistry == {}
    assert _warnings._defaultaction == 'default'
    expected = [('default', None, DeprecationWarning, '__main__', 0),
                ('ignore', None, DeprecationWarning, None, 0),
                ('ignore', None, PendingDeprecationWarning, None, 0),
                ('ignore', None, ImportWarning, None, 0),
                ('ignore', None, ResourceWarning, None, 0)]
    assert expected == _warnings.filters

def test_warn():
    _warnings.warn("some message", DeprecationWarning)
    _warnings.warn("some message", Warning)
    _warnings.warn(("some message",1), Warning)

def test_use_builtin__warnings():
    """Check that the stdlib warnings.py module manages to import our
    _warnings module.  If something is missing, it doesn't, and silently
    continues.  Then things don't reliably work: either the
    functionality of the pure Python version is subtly different, or
    more likely we get confusion because of a half-imported _warnings.
    """
    assert not hasattr(warnings, '_filters_version')

def test_lineno():
    with warnings.catch_warnings(record=True) as w:
        _warnings.warn("some message", Warning)
        lineno = sys._getframe().f_lineno - 1 # the line above
        assert w[-1].lineno == lineno

def test_warn_explicit():
    _warnings.warn_explicit("some message", DeprecationWarning,
                            "<string>", 1, module_globals=globals())
    _warnings.warn_explicit("some message", Warning,
                            "<string>", 1, module_globals=globals())

def test_with_source():
    source = []
    with warnings.catch_warnings(record=True) as w:
        _warnings.warn("some message", Warning, source=source)
    assert w[0].source is source

def test_default_action():
    warnings.defaultaction = 'ignore'
    warnings.resetwarnings()
    with warnings.catch_warnings(record=True) as w:
        _warnings.warn_explicit("message", UserWarning, "<test>", 44,
                                registry={})
        assert len(w) == 0
    warnings.defaultaction = 'default'

def test_ignore():
    warnings.resetwarnings()
    with warnings.catch_warnings(record=True) as w:
        __warningregistry__ = {}
        warnings.filterwarnings("ignore", category=UserWarning)
        _warnings.warn_explicit("message", UserWarning, "<test>", 44,
                                registry=__warningregistry__)
        assert len(w) == 0
        assert list(__warningregistry__) == ['version']

def test_show_source_line():
    try:
        from test.warning_tests import inner
    except ImportError:
        skip('no test, -A on cpython?')
    # With showarning() missing, make sure that output is okay.
    saved = warnings.showwarning
    try:
        del warnings.showwarning

        stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            inner('test message')
            result = sys.stderr.getvalue()
        finally:
            sys.stderr = stderr

        assert result.count('\n') == 2
        assert '  warnings.warn(message, ' in result
    finally:
        warnings.showwarning = saved

def test_filename_none():
    globals()['__file__'] = 'test.pyc'
    _warnings.warn('test', UserWarning)
    globals()['__file__'] = None
    _warnings.warn('test', UserWarning)

def test_bad_category():
    raises(TypeError, _warnings.warn, "text", 123)
    class Foo:
        pass
    raises(TypeError, _warnings.warn, "text", Foo)

def test_surrogate_in_filename():
    for filename in ("nonascii\xe9\u20ac", "surrogate\udc80"):
        try:
            __pypy__.fsencode(filename)
        except UnicodeEncodeError:
            continue
        _warnings.warn_explicit("text", UserWarning, filename, 1)

def test_issue31285():
    def get_bad_loader(splitlines_ret_val):
        class BadLoader:
            def get_source(self, fullname):
                class BadSource(str):
                    def splitlines(self):
                        return splitlines_ret_val
                return BadSource('spam')
        return BadLoader()
    # does not raise:
    _warnings.warn_explicit(
        'eggs', UserWarning, 'bar', 1,
        module_globals={'__loader__': get_bad_loader(42),
                        '__name__': 'foobar'})