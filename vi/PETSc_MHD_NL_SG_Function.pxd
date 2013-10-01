'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport DMDA, Mat, Vec

from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScFunction(object):
    '''
    
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef np.float64_t ht_inv
    cdef np.float64_t hx_inv
    cdef np.float64_t hy_inv
    
    cdef np.float64_t eps
    
    
    cdef DMDA da1
    cdef DMDA da5
    
    cdef Vec divV
    cdef Vec V
    cdef Vec Xh
    
    cdef Vec localV
    cdef Vec localB
    cdef Vec localX
    cdef Vec localXh
    
    cdef PETSc_MHD_Derivatives derivatives
    

    cdef np.float64_t dt_x(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dt_y(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)

    cdef np.float64_t curl(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.ndarray[np.float64_t, ndim=2] y,
                                 np.uint64_t i, np.uint64_t j)

    cdef np.float64_t ave_xt(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] xh,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t ave_yt(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] xh,
                                   np.uint64_t i, np.uint64_t j)
