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
CARGS        = ['-O3', '-axavx', '-march=corei7-avx', '-std=c99', '-Wno-unused-function', '-Wno-unneeded-internal-declaration']
LARGS        = []

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

# Intel MPI
IMPI_DIR = '/afs/@cell/common/soft/intel/ics2013/impi/4.1.3/intel64'

if isdir(IMPI_DIR):
    INCLUDE_DIRS += [join(IMPI_DIR, 'include')]
    LIBRARY_DIRS += [join(IMPI_DIR, 'lib')]

# OpenMPI
if isdir('/opt/local/include/openmpi-gcc5'):
    INCLUDE_DIRS += ['/opt/local/include/openmpi-gcc5']
if isdir('/opt/local/lib/openmpi-gcc5'):
    LIBRARY_DIRS += ['/opt/local/lib/openmpi-gcc5']

# MPI library
LIBRARIES    += ['mpi']

# Valgrind
INCLUDE_DIRS += ['/opt/local/include']
LIBRARY_DIRS += ['/opt/local/lib']


ext_modules = [
        Extension("PETSc_MHD_Derivatives",
                  sources=["PETSc_MHD_Derivatives.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("Ideal_MHD_NL_Matrix",
                  sources=["Ideal_MHD_NL_Matrix.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("Ideal_MHD_NL_Function",
                  sources=["Ideal_MHD_NL_Function.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("Ideal_MHD_NL_Jacobian_Matrix",
                  sources=["Ideal_MHD_NL_Jacobian_Matrix.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("Inertial_MHD_NL_Matrix",
                  sources=["Inertial_MHD_NL_Matrix.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("Inertial_MHD_NL_Function",
                  sources=["Inertial_MHD_NL_Function.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("Inertial_MHD_NL_Jacobian_Matrix",
                  sources=["Inertial_MHD_NL_Jacobian_Matrix.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
              ]
                
setup(
    name = 'PETSc Matrix-Free MHD Solver',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)