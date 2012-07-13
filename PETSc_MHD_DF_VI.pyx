'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, Mat, Vec

from PETSc_MHD_Derivatives import  PETSc_MHD_Derivatives
from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScSolver(object):
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
        self.S    = self.da4.createGlobalVec()
        self.divV = self.da1.createGlobalVec()
        
        # create history vectors
        self.Xh1 = self.da4.createGlobalVec()
        self.Xh2 = self.da4.createGlobalVec()
        
        # create local vectors
        self.localS   = da4.createLocalVec()
        self.localB   = da4.createLocalVec()
        self.localX   = da4.createLocalVec()
        self.localXh1 = da4.createLocalVec()
        self.localXh2 = da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        x   = self.da4.getVecArray(X)
        xh1 = self.da4.getVecArray(self.Xh1)
        xh2 = self.da4.getVecArray(self.Xh2)
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        xh2[xs:xe, ys:ye, :] = xh1[xs:xe, ys:ye, :]
        xh1[xs:xe, ys:ye, :] = x  [xs:xe, ys:ye, :]
        
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        cdef np.float64_t meanDivV
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(X,        self.localX)
        self.da4.globalToLocal(self.Xh1, self.localXh1)
        
        x  = self.da4.getVecArray(self.localX)
        xh = self.da4.getVecArray(self.localXh1)
        
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
        
        cdef np.ndarray[np.float64_t, ndim=2] divV = self.da1.getVecArray(self.divV)[...] 
        
        
        s = self.da4.getVecArray(self.S)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] tBx = s[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] tBy = s[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] tVx = s[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] tVy = s[:,:,3]
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                tBx[iy, jy] = \
                             + 0.5 * self.derivatives.dy1(Bx,  Vyh, ix, jx) \
                             + 0.5 * self.derivatives.dy1(Bxh, Vy,  ix, jx) \
                             - 0.5 * self.derivatives.dy1(By,  Vxh, ix, jx) \
                             - 0.5 * self.derivatives.dy1(Byh, Vx,  ix, jx)
                
                tBy[iy, jy] = \
                             + 0.2 * self.derivatives.dx1(By,  Vxh, ix, jx) \
                             + 0.2 * self.derivatives.dx1(Byh, Vx,  ix, jx) \
                             - 0.2 * self.derivatives.dx1(Bx,  Vyh, ix, jx) \
                             - 0.2 * self.derivatives.dx1(Bxh, Vy,  ix, jx)
                
                tVx[iy, jy] = \
                             + 0.5 * self.derivatives.dx1(Vx,  Vxh, ix, jx) \
                             + 0.5 * self.derivatives.dx1(Vxh, Vx,  ix, jx) \
                             + 0.5 * self.derivatives.dy1(Vx,  Vyh, ix, jx) \
                             + 0.5 * self.derivatives.dy1(Vxh, Vy,  ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bx,  Bxh, ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bxh, Bx,  ix, jx) \
                             - 0.5 * self.derivatives.dy1(Bx,  Byh, ix, jx) \
                             - 0.5 * self.derivatives.dy1(Bxh, By,  ix, jx)
                
                tVy[iy, jy] = \
                             + 0.5 * self.derivatives.dx1(Vx,  Vyh, ix, jx) \
                             + 0.5 * self.derivatives.dx1(Vxh, Vy,  ix, jx) \
                             + 0.5 * self.derivatives.dy1(Vy,  Vyh, ix, jx) \
                             + 0.5 * self.derivatives.dy1(Vyh, Vy,  ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bx,  Byh, ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bxh, By,  ix, jx) \
                             - 0.5 * self.derivatives.dy1(By,  Byh, ix, jx) \
                             - 0.5 * self.derivatives.dy1(Byh, By,  ix, jx)
                
        self.da4.globalToLocal(self.S, self.localS)
        
        s = self.da4.getVecArray(self.localS)[...]
        
        tBx = s[:,:,0]
        tBy = s[:,:,1]
        tVx = s[:,:,2]
        tVy = s[:,:,3]
                
        
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
                y[iy, jy, 0] = 0.5 * self.derivatives.dt(Bx, ix, jx) \
                             + 0.5 * tBx[ix, jx]
                    
                # B_y
                y[iy, jy, 1] = 0.5 * self.derivatives.dt(By, ix, jx) \
                             + 0.5 * tBy[ix, jx]
                
                # V_x
                y[iy, jy, 2] = 0.5 * self.derivatives.dt(Vx, ix, jx) \
                             + 0.5 * tVx[ix, jx] \
                             + 0.25 * self.derivatives.gradx(P, ix, jx)
                              
                # V_y
                y[iy, jy, 3] = 0.5 * self.derivatives.dt(Vy, ix, jx) \
                             + 0.5 * tVy[ix, jx] \
                             + 0.25 * self.derivatives.grady(P, ix, jx)
                              
                # P
                y[iy, jy, 4] = self.derivatives.laplace(P, ix, jx) \
                             + divV[iy, jy] - meanDivV
        
        
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(self.Xh1, self.localXh1)
        self.da4.globalToLocal(self.Xh2, self.localXh2)
        
        xh1 = self.da4.getVecArray(self.localXh1)
        xh2 = self.da4.getVecArray(self.localXh2)
        
        cdef np.ndarray[np.float64_t, ndim=3] b = self.da4.getVecArray(B)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh1 = xh1[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh1 = xh1[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh1 = xh1[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh1 = xh1[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] Ph1  = xh1[...][:,:,4]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh2 = xh2[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh2 = xh2[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh2 = xh2[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh2 = xh2[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] Ph2  = xh2[...][:,:,4]
        
        cdef np.ndarray[np.float64_t, ndim=2] divV = self.da1.getVecArray(self.divV)[...] 
        
        
        s = self.da4.getVecArray(self.S)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] tBx = s[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] tBy = s[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] tVx = s[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] tVy = s[:,:,3]
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                tBx[iy, jy] = \
                             + 0.5 * self.derivatives.dy1(Bxh1, Vyh2, ix, jx) \
                             + 0.5 * self.derivatives.dy1(Bxh2, Vyh1, ix, jx) \
                             - 0.5 * self.derivatives.dy1(Byh1, Vxh2, ix, jx) \
                             - 0.5 * self.derivatives.dy1(Byh2, Vxh1, ix, jx)
                
                tBx[iy, jy] = \
                             + 0.5 * self.derivatives.dx1(Byh1, Vxh2, ix, jx) \
                             + 0.5 * self.derivatives.dx1(Byh2, Vxh1, ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bxh1, Vyh2, ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bxh2, Vyh1, ix, jx)
                
                tVx[iy, jy] = \
                             + 0.5 * self.derivatives.dx1(Vxh1, Vxh2, ix, jx) \
                             + 0.5 * self.derivatives.dx1(Vxh2, Vxh1, ix, jx) \
                             + 0.5 * self.derivatives.dy1(Vxh1, Vyh2, ix, jx) \
                             + 0.5 * self.derivatives.dy1(Vxh2, Vyh1, ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bxh1, Bxh2, ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bxh2, Bxh1, ix, jx) \
                             - 0.5 * self.derivatives.dy1(Bxh1, Byh2, ix, jx) \
                             - 0.5 * self.derivatives.dy1(Bxh2, Byh1, ix, jx)
                
                tVy[iy, jy] = \
                             + 0.5 * self.derivatives.dx1(Vxh1, Vyh2, ix, jx) \
                             + 0.5 * self.derivatives.dx1(Vxh2, Vyh1, ix, jx) \
                             + 0.5 * self.derivatives.dy1(Vyh1, Vyh2, ix, jx) \
                             + 0.5 * self.derivatives.dy1(Vyh2, Vyh1, ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bxh1, Byh2, ix, jx) \
                             - 0.5 * self.derivatives.dx1(Bxh2, Byh1, ix, jx) \
                             - 0.5 * self.derivatives.dy1(Byh1, Byh2, ix, jx) \
                             - 0.5 * self.derivatives.dy1(Byh2, Byh1, ix, jx)
                
        self.da4.globalToLocal(self.S, self.localS)
        
        s = self.da4.getVecArray(self.localS)[...]
        
        tBx = s[:,:,0]
        tBy = s[:,:,1]        
        tVx = s[:,:,2]
        tVy = s[:,:,3]        
        
        
#        for i in np.arange(xs, xe):
#            ix = i-xs+1
#            iy = i-xs
#            
#            for j in np.arange(ys, ye):
#                jx = j-ys+1
#                jy = j-ys
#                
#                divV[iy, jy] = self.derivatives.gradx(tVx, ix, jx) \
#                             + self.derivatives.grady(tVy, ix, jx)
#                
#                
#        meanDivV = self.divV.sum() / (self.nx * self.ny)
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                b[iy, jy, 0] = 0.5 * self.derivatives.dt(Bxh2, ix, jx) \
                             - 0.5 * tBx[ix, jx]
                
                b[iy, jy, 1] = 0.5 * self.derivatives.dt(Byh2, ix, jx) \
                             - 0.5 * tBy[ix, jx]
                
                b[iy, jy, 2] = 0.5  * self.derivatives.dt(Vxh2, ix, jx) \
                             - 0.5  * tVx[ix, jx] \
                             - 0.5  * self.derivatives.gradx(Ph1, ix, jx) \
                             - 0.25 * self.derivatives.gradx(Ph2, ix, jx)

                b[iy, jy, 3] = 0.5  * self.derivatives.dt(Vyh2, ix, jx) \
                             - 0.5  * tVy[ix, jx] \
                             - 0.5  * self.derivatives.grady(Ph1, ix, jx) \
                             - 0.25 * self.derivatives.grady(Ph2, ix, jx)
                  
                b[iy, jy, 4] = 0.0 
    
    
    
#    @cython.boundscheck(False)
    def formRHSPoisson(self, Vec B, Vec X):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da4.globalToLocal(X, self.localX)
        
        x = self.da4.getVecArray(self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=2] b = self.da1.getVecArray(B)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By = x[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] divV = self.da1.getVecArray(self.divV)[...] 
        
        s = self.da4.getVecArray(self.S)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] tVx = s[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] tVy = s[:,:,3]
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                tVx[iy, jy] = \
                             + self.derivatives.dx1(Vx, Vx, ix, jx) \
                             + self.derivatives.dy1(Vx, Vy, ix, jx) \
                             - self.derivatives.dx1(Bx, Bx, ix, jx) \
                             - self.derivatives.dy1(Bx, By, ix, jx) \
                
                tVy[iy, jy] = \
                             + self.derivatives.dx1(Vx, Vy, ix, jx) \
                             + self.derivatives.dy1(Vy, Vy, ix, jx) \
                             - self.derivatives.dx1(Bx, By, ix, jx) \
                             - self.derivatives.dy1(By, By, ix, jx) \
                
        self.da4.globalToLocal(self.S, self.localS)
        
        s = self.da4.getVecArray(self.localS)[...]
        
        tVx = s[:,:,2]
        tVy = s[:,:,3]
        
        
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
                
                b[iy, jy] = - divV[iy, jy] + meanDivV
#                          - self.derivatives.gradx(tVx, ix, jx) \
#                          - self.derivatives.grady(tVy, ix, jx)


#    @cython.boundscheck(False)
    def timestep(self, np.ndarray[np.float64_t, ndim=3] x,
                       np.ndarray[np.float64_t, ndim=3] y):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By = x[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] P  = x[:,:,4]
        
        
        for j in np.arange(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
                # B_x
                y[iy, jy, 0] = \
                             - self.derivatives.dy1(Bx, Vy, ix, jx) \
                             + self.derivatives.dy1(By, Vx, ix, jx)
                    
                # B_y
                y[iy, jy, 1] = \
                             - self.derivatives.dx1(By, Vx, ix, jx) \
                             + self.derivatives.dx1(Bx, Vy, ix, jx)
                                
                # V_x
                y[iy, jy, 2] = \
                             - self.derivatives.dx1(Vx, Vx, ix, jx) \
                             - self.derivatives.dy1(Vx, Vy, ix, jx) \
                             + self.derivatives.dx1(Bx, Bx, ix, jx) \
                             + self.derivatives.dy1(Bx, By, ix, jx) \
                             - self.derivatives.gradx(P, ix, jx)
                              
                # V_y
                y[iy, jy, 3] = \
                             - self.derivatives.dx1(Vx, Vy, ix, jx) \
                             - self.derivatives.dy1(Vy, Vy, ix, jx) \
                             + self.derivatives.dx1(Bx, By, ix, jx) \
                             + self.derivatives.dy1(By, By, ix, jx) \
                             - self.derivatives.grady(P, ix, jx)
                              
                # P
                y[iy, jy, 4] = 0.0
          
