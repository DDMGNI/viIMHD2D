'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, Mat, Vec

from petsc_mhd_derivatives cimport PETSc_MHD_Derivatives



cdef class PETScSolver(object):
    '''
    
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    
    cdef DA da
    
    cdef Vec Xh1
    cdef Vec Xh2
    
    cdef Vec localB
    cdef Vec localX
    cdef Vec localXh1
    cdef Vec localXh2
    
    cdef np.ndarray ty
    
    cdef PETSc_MHD_Derivatives derivatives
  