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
    
    cdef np.float64_t fac_gradx
    cdef np.float64_t fac_grady
    cdef np.float64_t fac_divx
    cdef np.float64_t fac_divy
    cdef np.float64_t fac_dt
    cdef np.float64_t fac_fdudx
    cdef np.float64_t fac_fdudy
        
    cdef np.float64_t omega     # relaxation parameter
    
    
    cdef DMDA da1                 # distributed array controller for 1D data (pressure)
    cdef DMDA da4                 # distributed array controller for 4D data (velocity and magnetic field)
    
    cdef Vec Xh                 # last time step of V and B
#    cdef Vec Xp                 # last iteration of V and B
#    cdef Vec Ph                 # last time step of pressure
#    cdef Vec Pp                 # last iteration of pressure
    
    cdef Vec localX             # 
#    cdef Vec localP             # 
    cdef Vec localXh            # 
#    cdef Vec localPh            # 
#    cdef Vec localXp            # 
#    cdef Vec localPp            # 
    
    
    cdef PETSc_MHD_Derivatives derivatives
    
    
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] A,
                               np.uint64_t i, np.uint64_t j)
    
    cpdef np.float64_t fdudx(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign)

    cpdef np.float64_t fdudy(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign)

    cpdef np.float64_t udfdx(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign)

    cpdef np.float64_t udfdy(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign)

    
