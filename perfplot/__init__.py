# -*- coding: utf-8 -*-
#
from __future__ import print_function

from perfplot.__about__ import (
    __author__,
    __author_email__,
    __copyright__,
    __license__,
    __version__,
    __status__,
)

from perfplot.main import bench, plot, show, save

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__license__",
    "__version__",
    "__status__",
    "bench",
    "plot",
    "show",
    "save",
]

try:
    import pipdate
except ImportError:
    pass
else:
    if pipdate.needs_checking(__name__):
        print(pipdate.check(__name__, __version__), end="")
