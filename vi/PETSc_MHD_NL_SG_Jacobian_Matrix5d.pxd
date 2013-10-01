'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport DMDA, Mat, Vec

from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScJacobian(object):
    '''
    
    '''
    
    cdef np.uint64_t  nx        # no of grid points in x
    cdef np.uint64_t  ny        # no of grid points in y
    
    cdef np.float64_t ht        # step size in time
    cdef np.float64_t hx        # step size in x
    cdef np.float64_t hy        # step size in y
    
    cdef np.float64_t ht_inv
    cdef np.float64_t hx_inv
    cdef np.float64_t hy_inv
    
    cdef np.float64_t fac_dt
    
    cdef np.float64_t fac_divx
    cdef np.float64_t fac_divy
    
    cdef np.float64_t fac_gradx
    cdef np.float64_t fac_grady
        
    
    cdef DMDA da1                 # distributed array controller for 1D data
    cdef DMDA da5                 # distributed array controller for 5D data (velocity, magnetic field, pressure)
    
    cdef Vec Xh                 # last time step of V, B, p
    cdef Vec Xp                 # last iteration of V, B, p
    
    cdef Vec localXh            # 
    cdef Vec localXp            # 
    
    
    cdef PETSc_MHD_Derivatives derivatives
    
    
    cdef np.float64_t dt_x(self, np.ndarray[np.float64_t, ndim=2] A)
    
    cdef np.float64_t dt_y(self, np.ndarray[np.float64_t, ndim=2] A)
    
    cdef np.float64_t dudx(self, np.ndarray[np.float64_t, ndim=2] A,
                                 np.uint64_t i, np.uint64_t j,
                                 np.float64_t factor)

    cdef np.float64_t dudy(self, np.ndarray[np.float64_t, ndim=2] A,
                                 np.uint64_t i, np.uint64_t j,
                                 np.float64_t factor)

    cdef np.float64_t u_ave_xt(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t factor)

    cdef np.float64_t u_ave_yt(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t factor)

    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j)

    cdef np.float64_t ave_xt(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] xh,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t ave_yt(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] xh,
                                   np.uint64_t i, np.uint64_t j)
    
    
