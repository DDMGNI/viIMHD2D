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
        
        cdef np.ndarray[np.float64_t, ndim=3] tx
        cdef np.ndarray[np.float64_t, ndim=3] tx1
        cdef np.ndarray[np.float64_t, ndim=3] tx2
        cdef np.ndarray[np.float64_t, ndim=3] tx3
        cdef np.ndarray[np.float64_t, ndim=3] tx4
        
        
        self.da.globalToLocal(X, self.localX)
        tx  = self.da.getVecArray(self.localX)[...]
        tx1 = self.da.getVecArray(self.X1)[...]
        self.timestep(tx, tx1)
        
        self.da.globalToLocal(self.X1, self.localX1)
        tx1 = self.da.getVecArray(self.localX1)[...]
        tx2 = self.da.getVecArray(self.X2)[...]
        self.timestep(tx + 0.5 * self.ht * tx1, tx2)
        
        self.da.globalToLocal(self.X2, self.localX2)
        tx2 = self.da.getVecArray(self.localX2)[...]
        tx3 = self.da.getVecArray(self.X3)[...]
        self.timestep(tx + 0.5 * self.ht * tx2, tx3)
        
        self.da.globalToLocal(self.X3, self.localX3)
        tx3 = self.da.getVecArray(self.localX3)[...]
        tx4 = self.da.getVecArray(self.X4)[...]
        self.timestep(tx + 1.0 * self.ht * tx3, tx4)
        
        tx  = self.da.getVecArray(X)[...]
        tx1 = self.da.getVecArray(self.X1)[...]
        tx2 = self.da.getVecArray(self.X2)[...]
        tx3 = self.da.getVecArray(self.X3)[...]
        tx4 = self.da.getVecArray(self.X4)[...]
        
        tx[:,:,:] = tx + (tx1 + 2.*tx2 + 2.*tx3 + tx4) * self.ht / 6.

    
    
#    @cython.boundscheck(False)
    cdef timestep(self, np.ndarray[np.float64_t, ndim=3] tx,
                        np.ndarray[np.float64_t, ndim=3] ty):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] Bx = tx[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By = tx[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx = tx[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = tx[:,:,3]
         
        
        for j in np.arange(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
#                # B_x
#                ty[iy, jy, 0] = self.derivatives.dy(By, Vx, ix, jx) \
#                              - self.derivatives.dy(Bx, Vy, ix, jx)
#                    
#                # B_y
#                ty[iy, jy, 1] = self.derivatives.dx(Bx, Vy, ix, jx) \
#                              - self.derivatives.dx(By, Vx, ix, jx)
#                
#                # V_x
#                ty[iy, jy, 2] = self.derivatives.dy(Bx, By, ix, jx) \
#                              - self.derivatives.dy(Vx, Vy, ix, jx) \
#                              - self.derivatives.dx(Vx, Vx, ix, jx) \
#                              + 0.5 * self.derivatives.dx(Bx, Bx, ix, jx) \
#                              - 0.5 * self.derivatives.dx(By, By, ix, jx)
#                    
#                # V_y
#                ty[iy, jy, 3] = self.derivatives.dx(Bx, By, ix, jx) \
#                              - self.derivatives.dx(Vx, Vy, ix, jx) \
#                              - self.derivatives.dy(Vy, Vy, ix, jx) \
#                              - 0.5 * self.derivatives.dy(Bx, Bx, ix, jx) \
#                              + 0.5 * self.derivatives.dy(By, By, ix, jx)
    

                # B_x
                ty[iy, jy, 0] = self.derivatives.dy1(By, Vx, ix, jx) \
                              - self.derivatives.dy1(Bx, Vy, ix, jx)
                    
                # B_y
                ty[iy, jy, 1] = self.derivatives.dx1(Bx, Vy, ix, jx) \
                              - self.derivatives.dx1(By, Vx, ix, jx)
                
                # V_x
                ty[iy, jy, 2] = self.derivatives.dy1(Bx, By, ix, jx) \
                              - self.derivatives.dy1(Vx, Vy, ix, jx) \
                              - self.derivatives.dx1(Vx, Vx, ix, jx) \
                              + 0.5 * self.derivatives.dx1(Bx, Bx, ix, jx) \
                              - 0.5 * self.derivatives.dx1(By, By, ix, jx)
                    
                # V_y
                ty[iy, jy, 3] = self.derivatives.dx1(Bx, By, ix, jx) \
                              - self.derivatives.dx1(Vx, Vy, ix, jx) \
                              - self.derivatives.dy1(Vy, Vy, ix, jx) \
                              - 0.5 * self.derivatives.dy1(Bx, Bx, ix, jx) \
                              + 0.5 * self.derivatives.dy1(By, By, ix, jx)
    
    
#                # B_x
#                ty[iy, jy, 0] = self.derivatives.dy3(By, Vx, ix, jx) \
#                              - self.derivatives.dy3(Bx, Vy, ix, jx)
#                    
#                # B_y
#                ty[iy, jy, 1] = self.derivatives.dx3(Bx, Vy, ix, jx) \
#                              - self.derivatives.dx3(By, Vx, ix, jx)
#                
#                # V_x
#                ty[iy, jy, 2] = self.derivatives.dy3(Bx, By, ix, jx) \
#                              - self.derivatives.dy3(Vx, Vy, ix, jx) \
#                              - self.derivatives.dx3(Vx, Vx, ix, jx) \
#                              + 0.5 * self.derivatives.dx3(Bx, Bx, ix, jx) \
#                              - 0.5 * self.derivatives.dx3(By, By, ix, jx)
#                    
#                # V_y
#                ty[iy, jy, 3] = self.derivatives.dx3(Bx, By, ix, jx) \
#                              - self.derivatives.dx3(Vx, Vy, ix, jx) \
#                              - self.derivatives.dy3(Vy, Vy, ix, jx) \
#                              - 0.5 * self.derivatives.dy3(Bx, Bx, ix, jx) \
#                              + 0.5 * self.derivatives.dy3(By, By, ix, jx)
    

