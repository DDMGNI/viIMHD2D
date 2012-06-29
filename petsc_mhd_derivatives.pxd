'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport DA


cdef class PETSc_MHD_Derivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef DA da
    
    
        
    cdef np.float64_t dx1(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dx2(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dx3(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dx4(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)
    
    
    cdef np.float64_t dy1(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dy2(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dy3(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dy4(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)


    cdef np.float64_t gradx(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t grady(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)


    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j)


    cdef timestep(self, np.ndarray[np.float64_t, ndim=3] tx,
                        np.ndarray[np.float64_t, ndim=3] ty)

