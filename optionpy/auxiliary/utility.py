# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 11.07.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import warnings

from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

use_jit = True
force_nopython = False
use_cache = True


def maybe_jit(*jit_args, **jit_kwargs):
    # global use_jit, force_nopython, use_cache
    jit_kwargs.update({"nopython": False})

    if force_nopython:
        jit_kwargs.update({"nopython": True})
    if use_cache:
        jit_kwargs.update({"cache": True})

    def wrapper(fun):
        if use_jit:
            return jit(*jit_args, **jit_kwargs)(fun)
        else:
            return fun

    return wrapper
