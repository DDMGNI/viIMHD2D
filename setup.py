#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext

import os
from os.path import join, isdir

INCLUDE_DIRS = []
LIBRARY_DIRS = []
LIBRARIES    = []

# PETSc
PETSC_DIR  = os.environ['PETSC_DIR']
PETSC_ARCH = os.environ.get('PETSC_ARCH', '')

if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
    INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                     join(PETSC_DIR, 'include')]
    LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
else:
    if PETSC_ARCH: pass # XXX should warn ...
    INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
    LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]

LIBRARIES    += ['petsc']

# NumPy
import numpy
INCLUDE_DIRS += [numpy.get_include()]

# PETSc for Python
import petsc4py
INCLUDE_DIRS += [petsc4py.get_include()]

# MPI
INCLUDE_DIRS += ['/opt/local/include/openmpi']


ext_modules = [
        Extension("PETSc_MHD_Derivatives",
                  sources=["petsc_mhd_derivatives.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETSc_MHD_RK4",
                  sources=["petsc_mhd_rk4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETSc_MHD_VI",
                  sources=["petsc_mhd_vi.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 )
              ]
                
setup(
    name = 'PETSc Matrix-Free MHD Solver',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
