'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py cimport PETSc

from petsc4py.PETSc cimport Mat, Vec

from imhd.integrators.MHD_Derivatives cimport MHD_Derivatives



cdef class PETScFunction(object):
    '''
    
    '''
    
    cdef int  nx          # no of grid points in x
    cdef int  ny          # no of grid points in y
    
    cdef double ht        # step size in time
    cdef double hx        # step size in x
    cdef double hy        # step size in y
    
    cdef double ht_inv

    cdef double fac_grdx
    cdef double fac_grdy
    cdef double fac_divx
    cdef double fac_divy
    
    cdef double mu
    cdef double nu
    cdef double eta
    
    
    cdef object da1             # distributed array controller for 1D data
    cdef object da7             # distributed array controller for 5D data (velocity, magnetic field, pressure)
    
    cdef Vec divV
    cdef Vec V
    cdef Vec Xh                 # last time step of V, B, p
    cdef Vec Xp                 # last iteration of V, B, p
    
    cdef Vec localV
    cdef Vec localB
    cdef Vec localX
    cdef Vec localXh            # 
    cdef Vec localXp            # 
    
    cdef MHD_Derivatives derivatives

    
    cdef double muu(self, double[:,:] A, int i, int j)
    cdef double dt(self, double[:,:] A, int i, int j)
    cdef double rot(self, double[:,:] Ux, double[:,:] Uy, int i, int j)

    cdef double psix_ux(self, double[:,:] A, double[:,:] Ux, double[:,:] Uy, double[:,:] Vx, double[:,:] Vy, int i, int j, double sign)
    cdef double psix_uy(self, double[:,:] A, double[:,:] Ux, double[:,:] Uy, double[:,:] Vx, double[:,:] Vy, int i, int j, double sign)
    cdef double psiy_ux(self, double[:,:] A, double[:,:] Ux, double[:,:] Uy, double[:,:] Vx, double[:,:] Vy, int i, int j, double sign)
    cdef double psiy_uy(self, double[:,:] A, double[:,:] Ux, double[:,:] Uy, double[:,:] Vx, double[:,:] Vy, int i, int j, double sign)

    cdef double phix_ux(self, double[:,:] A, double[:,:] F, int i, int j, double sign)
    cdef double phix_uy(self, double[:,:] A, double[:,:] F, int i, int j, double sign)
    cdef double phiy_ux(self, double[:,:] A, double[:,:] F, int i, int j, double sign)
    cdef double phiy_uy(self, double[:,:] A, double[:,:] F, int i, int j, double sign)
