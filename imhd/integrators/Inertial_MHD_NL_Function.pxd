'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport Mat, Vec

from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScFunction(object):
    '''
    
    '''
    
    cdef int  nx
    cdef int  ny
    
    cdef double ht
    cdef double hx
    cdef double hy
    
    cdef double ht_inv
    
    cdef double mu
    cdef double nu
    cdef double eta
    cdef double de
    
    
    cdef object da1
    cdef object da5
    
    cdef Vec divV
    cdef Vec V
    cdef Vec Xh
    
    cdef Vec localV
    cdef Vec localB
    cdef Vec localX
    cdef Vec localXh
    
    cdef PETSc_MHD_Derivatives derivatives
    
    
    cdef double dt(self, double[:,:] x, int i, int j)
    cdef double dt_x(self, double[:,:] x, int i, int j)
    cdef double dt_y(self, double[:,:] x, int i, int j)

