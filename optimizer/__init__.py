from os.path import dirname, basename, isfile, join
import sys
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
for mod in __all__:
  mod = __import__('.'.join([__name__, mod]), fromlist=[mod])
  to_import = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
  for i in to_import:
    try:
      setattr(sys.modules[__name__], i.__name__, i)
    except AttributeError:
      pass