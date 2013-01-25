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
    
    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j)

    cdef np.float64_t fdudx(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j)

    cdef np.float64_t fdudy(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t fx_dx_ux(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t fx_dx_uy(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t fy_dy_ux(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t fy_dy_uy(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dx_fx_uy(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dy_fx_uy(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j)
    

    cdef np.float64_t gradx(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t grady(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t gradx_sg(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.uint64_t i, np.uint64_t j)

    cdef np.float64_t grady_sg(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.uint64_t i, np.uint64_t j)

    cdef np.float64_t gradx_fv(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t grady_fv(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.uint64_t i, np.uint64_t j)

    cdef np.float64_t gradx_simple(self, np.ndarray[np.float64_t, ndim=2] x,
                                         np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t grady_simple(self, np.ndarray[np.float64_t, ndim=2] x,
                                         np.uint64_t i, np.uint64_t j)

    cdef np.float64_t divx(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t divy(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j)

    cdef np.float64_t divx_sg(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j)

    cdef np.float64_t divy_sg(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j)

    cdef np.float64_t laplace(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j)

    cdef np.float64_t laplace_fd(self, np.ndarray[np.float64_t, ndim=2] x,
                                       np.uint64_t i, np.uint64_t j)


    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dt_diag(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j)


    cdef np.float64_t rot(self, np.ndarray[np.float64_t, ndim=2] Ux,
                                np.ndarray[np.float64_t, ndim=2] Uy,
                                np.uint64_t i, np.uint64_t j)

    cdef np.float64_t psix(self, np.ndarray[np.float64_t, ndim=2] Ux,
                                 np.ndarray[np.float64_t, ndim=2] Uy,
                                 np.uint64_t i, np.uint64_t j)

    cdef np.float64_t psiy(self, np.ndarray[np.float64_t, ndim=2] Ux,
                                 np.ndarray[np.float64_t, ndim=2] Uy,
                                 np.uint64_t i, np.uint64_t j)

    cdef np.float64_t phix(self, np.ndarray[np.float64_t, ndim=2] F,
                                 np.ndarray[np.float64_t, ndim=2] U,
                                 np.uint64_t i, np.uint64_t j)

    cdef np.float64_t phiy(self, np.ndarray[np.float64_t, ndim=2] F,
                                 np.ndarray[np.float64_t, ndim=2] U,
                                 np.uint64_t i, np.uint64_t j)

