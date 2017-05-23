# -*- coding: utf-8 -*-
#
from perfplot.__about__ import (
        __author__,
        __author_email__,
        __copyright__,
        __license__,
        __version__,
        __status__
        )

from perfplot.main import *

import pipdated
if pipdated.needs_checking(__name__):
    print(pipdated.check(__name__, __version__))
