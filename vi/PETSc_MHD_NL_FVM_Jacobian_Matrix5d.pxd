'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, Mat, Vec

from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScJacobian(object):
    '''
    
    '''
    
    cdef np.uint64_t  nx        # no of grid points in x
    cdef np.uint64_t  ny        # no of grid points in y
    
    cdef np.float64_t ht        # step size in time
    cdef np.float64_t hx        # step size in x
    cdef np.float64_t hy        # step size in y
    
    cdef np.float64_t fac_dt
    cdef np.float64_t fac_dx
    cdef np.float64_t fac_dy
    
    cdef np.float64_t fac_grdx
    cdef np.float64_t fac_grdy
    cdef np.float64_t fac_divx
    cdef np.float64_t fac_divy
    
    
    cdef DA da1                 # distributed array controller for 1D data
    cdef DA da5                 # distributed array controller for 5D data (velocity, magnetic field, pressure)
    
    cdef Vec Xh                 # last time step of V, B, p
    cdef Vec Xp                 # last iteration of V, B, p
    
    cdef Vec localXh            # 
    cdef Vec localXp            # 
    
    
    cdef PETSc_MHD_Derivatives derivatives
    
    
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] A,
                               np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dt_x(self, np.ndarray[np.float64_t, ndim=2] A,
                                 np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dt_y(self, np.ndarray[np.float64_t, ndim=2] A,
                                 np.uint64_t i, np.uint64_t j)
    

    cdef np.float64_t rot(self, np.ndarray[np.float64_t, ndim=2] Ux,
                                np.ndarray[np.float64_t, ndim=2] Uy,
                                np.uint64_t i, np.uint64_t j)


    cdef np.float64_t psix_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Ux,
                                    np.ndarray[np.float64_t, ndim=2] Uy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign)

    cdef np.float64_t psix_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Ux,
                                    np.ndarray[np.float64_t, ndim=2] Uy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign)

    cdef np.float64_t psiy_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Ux,
                                    np.ndarray[np.float64_t, ndim=2] Uy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign)

    cdef np.float64_t psiy_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Ux,
                                    np.ndarray[np.float64_t, ndim=2] Uy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign)


    cdef np.float64_t phix_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] F,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign)

    cdef np.float64_t phix_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] F,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign)

    cdef np.float64_t phiy_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] F,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign)

    cdef np.float64_t phiy_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] F,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign)



