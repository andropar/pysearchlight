"""PySearchlight package exports the :class:`SearchLight` class."""

# The main entry point of this package is the ``SearchLight`` class defined in
# ``sl.py``.  The class name uses a capital "L", but an earlier version of this
# file tried to import ``Searchlight`` (lowercase "l"), which resulted in an
# ``ImportError`` when importing the package.  Import the correct class name so
# ``from pysearchlight import SearchLight`` works as expected.

from .sl import SearchLight

__all__ = ["SearchLight"]
