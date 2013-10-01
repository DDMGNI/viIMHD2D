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
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    
    cdef DMDA da1
    cdef DMDA da4
    
    cdef Vec Xh
    cdef Vec Xp
    
    cdef Vec localX
    cdef Vec localXh
    cdef Vec localXp
    
    cdef PETSc_MHD_Derivatives derivatives
    
