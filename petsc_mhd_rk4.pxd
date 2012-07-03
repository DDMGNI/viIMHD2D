'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport DA, Vec

from petsc_mhd_derivatives cimport PETSc_MHD_Derivatives 


cdef class PETScRK4(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef DA da
    
    cdef Vec X1
    cdef Vec X2
    cdef Vec X3
    cdef Vec X4
    
    cdef Vec localX
    cdef Vec localX1
    cdef Vec localX2
    cdef Vec localX3
    cdef Vec localX4
    
    cdef PETSc_MHD_Derivatives derivatives
    
        
