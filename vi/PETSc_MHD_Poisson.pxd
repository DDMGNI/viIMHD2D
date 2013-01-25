'''
Created on Jul 04, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives


cdef class PETScPoissonSolver(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef DA da1
    cdef DA da4
    
    cdef Vec X
    cdef Vec V
    
    cdef Vec localV
    cdef Vec localP
    cdef Vec localX
    
    cdef PETSc_MHD_Derivatives derivatives
