'''
Created on Jul 04, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec


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
#    cdef Vec localU
    cdef Vec localX
    

    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dx1(self, np.ndarray[np.float64_t, ndim=2] x,
                                np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dy1(self, np.ndarray[np.float64_t, ndim=2] x,
                                np.uint64_t i, np.uint64_t j)

    cdef np.float64_t laplace(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j)
