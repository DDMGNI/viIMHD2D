'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np


cdef class PETSc_MHD_Derivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef np.float64_t ht_inv
    cdef np.float64_t hx_inv
    cdef np.float64_t hy_inv
    
    cdef np.float64_t hx_inv2
    cdef np.float64_t hy_inv2
    
    
        
    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dx1(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t fdudx(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j)

    cdef np.float64_t fdudx_diag(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j)

    cdef np.float64_t fdudx_diag_fac(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dx3(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)

    
    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dy1(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)

    cdef np.float64_t fdudy(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t fdudy_diag(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t fdudy_diag_fac(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dy3(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j)
    

    cdef np.float64_t gradx(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t grady(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)

    cdef np.float64_t laplace(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j)


    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dt_diag(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dt_diag_fac(self)
    
    cdef np.float64_t dt_diag_inv(self, np.ndarray[np.float64_t, ndim=2] x,
                                        np.uint64_t i, np.uint64_t j)
