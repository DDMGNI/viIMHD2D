'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport Mat, Vec

from imhd.integrators.MHD_Derivatives cimport MHD_Derivatives



cdef class PETScSolverPoisson(object):
    '''
    
    '''
    
    cdef int  nx          # no of grid points in x
    cdef int  ny          # no of grid points in y
    
    cdef double ht        # step size in time
    cdef double hx        # step size in x
    cdef double hy        # step size in y
    
    cdef double mu
    cdef double nu
    cdef double eta
    cdef double de
    
    cdef object da2             # distributed array controller for 1D data
    cdef object da7             # distributed array controller for 5D data (velocity, magnetic field, pressure)
    
    cdef Vec Bp
    cdef Vec Xh                 # last time step of V, B, p
    cdef Vec Xp                 # last iteration of V, B, p
    
    cdef Vec localB
    cdef Vec localX
    cdef Vec localXh            # 
    cdef Vec localXp            # 
    
    cdef MHD_Derivatives derivatives
    