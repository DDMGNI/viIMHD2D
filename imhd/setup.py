#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext

import os
from os.path import join, isdir

INCLUDE_DIRS = [os.curdir]
LIBRARY_DIRS = []
LIBRARIES    = []
CARGS        = ['-O3', '-std=c99', '-Wno-unused-function', '-Wno-unneeded-internal-declaration']
#CARGS        = ['-O3', '-axavx', '-march=corei7-avx', '-std=c99', '-Wno-unused-function', '-Wno-unneeded-internal-declaration']
LARGS        = []

# NumPy
import numpy
INCLUDE_DIRS += [numpy.get_include()]

ext_modules = [
        Extension("diagnostics",
                  sources=["diagnostics.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
              ]
                
setup(
    name = 'PETSc Ideal and Inertial MHD Diagnostics',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
