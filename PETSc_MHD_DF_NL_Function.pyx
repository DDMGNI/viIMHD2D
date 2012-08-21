'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, SNES, Mat, Vec

from PETSc_MHD_Derivatives import  PETSc_MHD_Derivatives
from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScFunction(object):
    '''
    
    '''
    
    def __init__(self, DA da1, DA da4,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        assert da1.getDim() == 2
        assert da4.getDim() == 2
        
        # distributed array
        self.da1 = da1
        self.da4 = da4
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        # create temporary vector
        self.V    = self.da4.createGlobalVec()
        self.divV = self.da1.createGlobalVec()
        
        # create history vector
        self.Xh = self.da4.createGlobalVec()
        
        # create local vectors
        self.localV  = da4.createLocalVec()
        self.localB  = da4.createLocalVec()
        self.localX  = da4.createLocalVec()
        self.localXh = da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        x  = self.da4.getVecArray(X)
        xh = self.da4.getVecArray(self.Xh)
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        xh[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
        
#    @cython.boundscheck(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        cdef np.float64_t meanDivV
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(X,       self.localX)
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        x  = self.da4.getVecArray(self.localX)
        xh = self.da4.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] y = self.da4.getVecArray(Y)[...]

        cdef np.ndarray[np.float64_t, ndim=2] Bx  = x [...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By  = x [...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx  = x [...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy  = x [...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] P   = x [...][:,:,4]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] Ph  = xh[...][:,:,4]
        
        cdef np.ndarray[np.float64_t, ndim=2] divV = self.da1.getVecArray(self.divV)[...] 
        
        
        v = self.da4.getVecArray(self.V)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] tVx = v[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] tVy = v[:,:,1]
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                tVx[iy, jy] = \
                             + 0.25 * self.derivatives.dx1(Vx,  Vx,  ix, jx) \
                             + 0.25 * self.derivatives.dx1(Vx,  Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Vxh, Vx,  ix, jx) \
                             + 0.25 * self.derivatives.dx1(Vxh, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Vx,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dy1(Vx,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Vxh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dy1(Vxh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bx,  Bx,  ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bx,  Bxh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bxh, Bx,  ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bxh, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Bx,  By,  ix, jx) \
                             - 0.25 * self.derivatives.dy1(Bx,  Byh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Bxh, By,  ix, jx) \
                             - 0.25 * self.derivatives.dy1(Bxh, Byh, ix, jx)
                
                tVy[iy, jy] = \
                             + 0.25 * self.derivatives.dx1(Vx,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dx1(Vx,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Vxh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dy1(Vy,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dy1(Vy,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Vyh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dx1(Vxh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Vyh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bx,  By,  ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bx,  Byh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bxh, By,  ix, jx) \
                             - 0.25 * self.derivatives.dy1(By,  By,  ix, jx) \
                             - 0.25 * self.derivatives.dy1(By,  Byh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Byh, By,  ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bxh, Byh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Byh, Byh, ix, jx)
                
        self.da4.globalToLocal(self.V, self.localV)
        
        v = self.da4.getVecArray(self.localV)[...]
        
        tVx = v[:,:,0]
        tVy = v[:,:,1]
                
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                divV[iy, jy] = self.derivatives.gradx(tVx, ix, jx) \
                             + self.derivatives.grady(tVy, ix, jx)
                
                
        meanDivV = self.divV.sum() / (self.nx * self.ny)
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                # B_x
                y[iy, jy, 0] = self.derivatives.dt(Bx,  ix, jx) \
                             - self.derivatives.dt(Bxh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Bx,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dy1(Bx,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Bxh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dy1(Bxh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(By,  Vx,  ix, jx) \
                             - 0.25 * self.derivatives.dy1(By,  Vxh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Byh, Vx,  ix, jx) \
                             - 0.25 * self.derivatives.dy1(Byh, Vxh, ix, jx)
                    
                # B_y
                y[iy, jy, 1] = self.derivatives.dt(By,  ix, jx) \
                             - self.derivatives.dt(Byh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(By,  Vx,  ix, jx) \
                             + 0.25 * self.derivatives.dx1(By,  Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Byh, Vx,  ix, jx) \
                             + 0.25 * self.derivatives.dx1(Byh, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bx,  Vy,  ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bx,  Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bxh, Vy,  ix, jx) \
                             - 0.25 * self.derivatives.dx1(Bxh, Vyh, ix, jx)
                
                # V_x
                y[iy, jy, 2] = self.derivatives.dt(Vx,  ix, jx) \
                             - self.derivatives.dt(Vxh, ix, jx) \
                             + tVx[ix, jx] \
                             + 0.5 * self.derivatives.gradx(P,  ix, jx) \
                             + 0.5 * self.derivatives.gradx(Ph, ix, jx)
                              
                # V_y
                y[iy, jy, 3] = self.derivatives.dt(Vy,  ix, jx) \
                             - self.derivatives.dt(Vyh, ix, jx) \
                             + tVy[ix, jx] \
                             + 0.5 * self.derivatives.grady(P,  ix, jx) \
                             + 0.5 * self.derivatives.grady(Ph, ix, jx)
                              
                # P
                y[iy, jy, 4] = self.derivatives.laplace(P, ix, jx) \
                             + divV[iy, jy] - meanDivV
        

