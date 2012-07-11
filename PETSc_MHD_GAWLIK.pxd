'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, Mat, Vec

from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScSolver(object):
    '''
    
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    
    cdef DA da
    
    cdef Vec P
    cdef Vec Xh
    
    cdef Vec localP
    cdef Vec localB
    cdef Vec localX
    cdef Vec localXh
    
    cdef Vec X1
    cdef Vec X2
    cdef Vec X3
    cdef Vec X4
    
    cdef Vec localX1
    cdef Vec localX2
    cdef Vec localX3
    cdef Vec localX4
    
    
    cdef PETSc_MHD_Derivatives derivatives
    
    
    cdef np.float64_t psi_x(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.uint64_t i, np.uint64_t j)

    cdef np.float64_t psi_y(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.uint64_t i, np.uint64_t j)

    cdef np.float64_t phi_x(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.ndarray[np.float64_t, ndim=2] Bx,
                                  np.ndarray[np.float64_t, ndim=2] By,
                                  np.uint64_t i, np.uint64_t j)
        
    cdef np.float64_t phi_y(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.ndarray[np.float64_t, ndim=2] Bx,
                                  np.ndarray[np.float64_t, ndim=2] By,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t omega(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t ave_x(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t ave_y(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j)
    

    cdef np.float64_t dt_xave(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dt_yave(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j)

    cdef timestep(self, np.ndarray[np.float64_t, ndim=3] tx,
                        np.ndarray[np.float64_t, ndim=3] ty)
