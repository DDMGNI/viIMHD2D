'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport DMDA, Mat, Vec

from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScPoisson(object):
    '''
    
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    
    cdef DMDA da1
    cdef DMDA da4
    
    cdef Vec dPdx
    cdef Vec dPdy
    
    cdef Vec localdPdx
    cdef Vec localdPdy
    
    cdef Vec localX
    cdef Vec localR
    
    cdef PETSc_MHD_Derivatives derivatives
    
