'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport DA, Vec

from PETSc_MHD_Derivatives import  PETSc_MHD_Derivatives
from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives


cdef class PETScRK4(object):
    '''
    PETSc/Cython Implementation of Explicit RK4 MHD Solver
    '''
    
    
    def __cinit__(self, DA da,
                  np.uint64_t nx, np.uint64_t ny,
                  np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy

        # distributed array
        self.da = da
        
        # create global vectors
        self.X1 = self.da.createGlobalVec()
        self.X2 = self.da.createGlobalVec()
        self.X3 = self.da.createGlobalVec()
        self.X4 = self.da.createGlobalVec()
        
        # create local vectors
        self.localX  = da.createLocalVec()
        self.localX1 = da.createLocalVec()
        self.localX2 = da.createLocalVec()
        self.localX3 = da.createLocalVec()
        self.localX4 = da.createLocalVec()
        
        # create derivative object
        self.derivatives = PETSc_MHD_Derivatives(da, nx, ny, ht, hx, hy)
        
     
    def rk4(self, Vec X):
        
        self.da.globalToLocal(X,       self.localX)
        self.da.globalToLocal(self.X1, self.localX1)
        self.da.globalToLocal(self.X2, self.localX2)
        self.da.globalToLocal(self.X3, self.localX3)
        self.da.globalToLocal(self.X4, self.localX4)
        
        x  = self.da.getVecArray(self.localX)
        x1 = self.da.getVecArray(self.localX1)
        x2 = self.da.getVecArray(self.localX2)
        x3 = self.da.getVecArray(self.localX3)
        x4 = self.da.getVecArray(self.localX4)
        
        cdef np.ndarray[np.float64_t, ndim=3] tx  = x [...]
        cdef np.ndarray[np.float64_t, ndim=3] tx1 = x1[...]
        cdef np.ndarray[np.float64_t, ndim=3] tx2 = x2[...]
        cdef np.ndarray[np.float64_t, ndim=3] tx3 = x3[...]
        cdef np.ndarray[np.float64_t, ndim=3] tx4 = x4[...]
        
        
        self.derivatives.mhd_timestep(tx, tx1)
        self.da.localToGlobal(self.localX1, self.X1)
        
        self.da.globalToLocal(self.X1, self.localX1); tx1 = x1[...]
        self.derivatives.mhd_timestep(tx + 0.5 * self.ht * tx1, tx2)
        self.da.localToGlobal(self.localX2, self.X2)
        
        self.da.globalToLocal(self.X2, self.localX2); tx2 = x2[...]
        self.derivatives.mhd_timestep(tx + 0.5 * self.ht * tx2, tx3)
        self.da.localToGlobal(self.localX3, self.X3)
        
        self.da.globalToLocal(self.X3, self.localX3); tx3 = x3[...]
        self.derivatives.mhd_timestep(tx + 1.0 * self.ht * tx3, tx4)
        self.da.localToGlobal(self.localX4, self.X4)
        
        tx  = self.da.getVecArray(X)[...]
        tx1 = self.da.getVecArray(self.X1)[...]
        tx2 = self.da.getVecArray(self.X2)[...]
        tx3 = self.da.getVecArray(self.X3)[...]
        tx4 = self.da.getVecArray(self.X4)[...]
        
        tx[:,:,:] += (tx1 + 2.*tx2 + 2.*tx3 + tx4) * self.ht / 6.
        
