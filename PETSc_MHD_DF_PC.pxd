'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, Mat, Vec

from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScPreconditioner(object):
    '''
    
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    
    cdef DA da1
    cdef DA da4
    
    cdef Vec P
    cdef Vec V
    cdef Vec Xh
    
    cdef Vec localP
    cdef Vec localV
    cdef Vec localB
    cdef Vec localX
    cdef Vec localXh
    
    cdef PETSc_MHD_Derivatives derivatives
    
