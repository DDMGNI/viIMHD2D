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
        self.V = self.da4.createGlobalVec()
        
        # create history vector
        self.Xh = self.da4.createGlobalVec()
        
        # create local vectors
        self.localV  = da4.createLocalVec()
        self.localB  = da4.createLocalVec()
        self.localX  = da4.createLocalVec()
        self.localXh = da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(da4, nx, ny, ht, hx, hy)
        
        
    
    def update_history(self, Vec X):
        x  = self.da4.getVecArray(X)
        xh = self.da4.getVecArray(self.Xh)
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        xh[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(X,       self.localX)
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        x  = self.da4.getVecArray(self.localX)
        xh = self.da4.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] ty  = self.da4.getVecArray(Y)[...]

        cdef np.ndarray[np.float64_t, ndim=2] Bx  = x [...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By  = x [...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx  = x [...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy  = x [...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] P   = x [...][:,:,4]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,3]
        
        
        v = self.da4.getVecArray(self.V)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] tVx = v[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] tVy = v[:,:,1]
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                tVx[iy, jy] = + 0.25 * self.dx(Vx,  Vx,  ix, jx) \
                              + 0.25 * self.dx(Vx,  Vxh, ix, jx) \
                              + 0.25 * self.dx(Vxh, Vx,  ix, jx) \
                              + 0.25 * self.dy(Vx,  Vy,  ix, jx) \
                              + 0.25 * self.dy(Vx,  Vyh, ix, jx) \
                              + 0.25 * self.dy(Vxh, Vy,  ix, jx) \
                              - 0.25 * self.dx(Bx,  Bx,  ix, jx) \
                              - 0.25 * self.dx(Bx,  Bxh, ix, jx) \
                              - 0.25 * self.dx(Bxh, Bx,  ix, jx) \
                              - 0.25 * self.dy(Bx,  By,  ix, jx) \
                              - 0.25 * self.dy(Bx,  Byh, ix, jx) \
                              - 0.25 * self.dy(Bxh, By,  ix, jx)
                
                tVy[iy, jy] = + 0.25 * self.dx(Vx,  Vy,  ix, jx) \
                              + 0.25 * self.dx(Vx,  Vyh, ix, jx) \
                              + 0.25 * self.dx(Vxh, Vy,  ix, jx) \
                              + 0.25 * self.dy(Vy,  Vy,  ix, jx) \
                              + 0.25 * self.dy(Vy,  Vyh, ix, jx) \
                              + 0.25 * self.dy(Vyh, Vy,  ix, jx) \
                              - 0.25 * self.dx(Bx,  By,  ix, jx) \
                              - 0.25 * self.dx(Bx,  Byh, ix, jx) \
                              - 0.25 * self.dx(Bxh, By,  ix, jx) \
                              - 0.25 * self.dy(By,  By,  ix, jx) \
                              - 0.25 * self.dy(By,  Byh, ix, jx) \
                              - 0.25 * self.dy(Byh, By,  ix, jx)
                
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
                
                    
                # B_x
                ty[iy, jy, 0] = self.derivatives.dt(Bx, ix, jx) \
                              + 0.25 * self.dy(Bx,  Vy,  ix, jx) \
                              + 0.25 * self.dy(Bxh, Vy,  ix, jx) \
                              + 0.25 * self.dy(Bx,  Vyh, ix, jx) \
                              - 0.25 * self.dy(By,  Vx,  ix, jx) \
                              - 0.25 * self.dy(Byh, Vx,  ix, jx) \
                              - 0.25 * self.dy(By,  Vxh, ix, jx)
                    
                # B_y
                ty[iy, jy, 1] = self.derivatives.dt(By, ix, jx) \
                              + 0.25 * self.dx(By,  Vx,  ix, jx) \
                              + 0.25 * self.dx(Byh, Vx,  ix, jx) \
                              + 0.25 * self.dx(By,  Vxh, ix, jx) \
                              - 0.25 * self.dx(Bx,  Vy,  ix, jx) \
                              - 0.25 * self.dx(Bxh, Vy,  ix, jx) \
                              - 0.25 * self.dx(Bx,  Vyh, ix, jx)
                
                # V_x
                ty[iy, jy, 2] = self.derivatives.dt(Vx, ix, jx) \
                              + tVx[iy, jy] \
                              + 0.5 * self.derivatives.gradx(P, ix, jx)
                    
                # V_y
                ty[iy, jy, 3] = self.derivatives.dt(Vy, ix, jx) \
                              + tVy[iy, jy] \
                              + 0.5 * self.derivatives.grady(P, ix, jx)
                
                # P
                ty[iy, jy, 4] = 0.5 * self.derivatives.laplace(P, ix, jx) \
                              + 0.5 * self.derivatives.gradx(tVx, ix, jx) \
                              + 0.5 * self.derivatives.grady(tVy, ix, jx)
                
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        xh = self.da4.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] tb  = self.da4.getVecArray(B)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] Ph  = xh[...][:,:,4]
        
        
        v = self.da4.getVecArray(self.V)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] tVx = v[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] tVy = v[:,:,1]
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                tVx[iy, jy] = + self.dx(Vxh, Vxh, ix, jx) \
                              + self.dy(Vxh, Vyh, ix, jx) \
                              - self.dx(Bxh, Bxh, ix, jx) \
                              - self.dy(Bxh, Byh, ix, jx)
                
                tVy[iy, jy] = + self.dx(Vxh, Vyh, ix, jx) \
                              + self.dy(Vyh, Vyh, ix, jx) \
                              - self.dx(Bxh, Byh, ix, jx) \
                              - self.dy(Byh, Byh, ix, jx)
                
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
                
                tb[iy, jy, 0] = self.derivatives.dt(Bxh, ix, jx) \
                              - 0.25 * self.dy(Bxh, Vyh, ix, jx) \
                              + 0.25 * self.dy(Byh, Vxh, ix, jx)
                
                tb[iy, jy, 1] = self.derivatives.dt(Byh, ix, jx) \
                              - 0.25 * self.dx(Byh, Vxh, ix, jx) \
                              + 0.25 * self.dx(Bxh, Vyh, ix, jx)
                
                tb[iy, jy, 2] = self.derivatives.dt(Vxh, ix, jx) \
                              - 0.25 * self.dx(Vxh, Vxh, ix, jx) \
                              - 0.25 * self.dy(Vxh, Vyh, ix, jx) \
                              + 0.25 * self.dx(Bxh, Bxh, ix, jx) \
                              + 0.25 * self.dy(Bxh, Byh, ix, jx) \
                              - 0.5 * self.derivatives.gradx(Ph, ix, jx)
                
                tb[iy, jy, 3] = self.derivatives.dt(Vyh, ix, jx) \
                              - 0.25 * self.dx(Vxh, Vyh, ix, jx) \
                              - 0.25 * self.dy(Vyh, Vyh, ix, jx) \
                              + 0.25 * self.dx(Bxh, Byh, ix, jx) \
                              + 0.25 * self.dy(Byh, Byh, ix, jx) \
                              - 0.5 * self.derivatives.grady(Ph, ix, jx)
                
                tb[iy, jy, 4] = 0.0
#                                0.5 * self.derivatives.laplace(Ph, ix, jx) \
#                              - 0.5 * self.derivatives.gradx(tVx, ix, jx) \
#                              - 0.5 * self.derivatives.grady(tVy, ix, jx)

                
        
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
                              - self.dx(Vx, Vx, ix, jx) \
                              - self.dy(Vx, Vy, ix, jx) \
                              + self.dx(Bx, Bx, ix, jx) \
                              + self.dy(Bx, By, ix, jx)
                
                tVy[iy, jy] = \
                              - self.dx(Vx, Vy, ix, jx) \
                              - self.dy(Vy, Vy, ix, jx) \
                              + self.dx(Bx, By, ix, jx) \
                              + self.dy(By, By, ix, jx)
                
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
                
                b[iy, jy] = \
                          + self.derivatives.gradx(tVx, ix, jx) \
                          + self.derivatives.grady(tVy, ix, jx)



#    @cython.boundscheck(False)
    def timestep(self, np.ndarray[np.float64_t, ndim=3] tx,
                       np.ndarray[np.float64_t, ndim=3] ty):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] Bx = tx[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By = tx[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx = tx[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = tx[:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] P  = tx[:,:,4]
         
        
        for j in np.arange(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
                ty[iy, jy, 0] = \
                              - self.dy(Bx, Vy, ix, jx) \
                              + self.dy(By, Vx, ix, jx)
                
                ty[iy, jy, 1] = \
                              - self.dx(By, Vx, ix, jx) \
                              + self.dx(Bx, Vy, ix, jx)
                
                ty[iy, jy, 2] = \
                              - self.dx(Vx, Vx, ix, jx) \
                              - self.dy(Vx, Vy, ix, jx) \
                              + self.dx(Bx, Bx, ix, jx) \
                              + self.dy(Bx, By, ix, jx) \
                              - self.derivatives.gradx(P, ix, jx)
                
                ty[iy, jy, 3] = \
                              - self.dx(Vx, Vy, ix, jx) \
                              - self.dy(Vy, Vy, ix, jx) \
                              + self.dx(Bx, By, ix, jx) \
                              + self.dy(By, By, ix, jx) \
                              - self.derivatives.grady(P, ix, jx)

                ty[iy, jy, 4] = 0.0
          
    

#    @cython.boundscheck(False)
    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dx centred finite differences
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                     + 1 * ( V[i+1, j-1] * B[i+1, j-1] - V[i-1, j-1] * B[i-1, j-1] ) \
                     + 2 * ( V[i+1, j  ] * B[i+1, j  ] - V[i-1, j  ] * B[i-1, j  ] ) \
                     + 1 * ( V[i+1, j+1] * B[i+1, j+1] - V[i-1, j+1] * B[i-1, j+1] ) \
                 ) / self.hx
 
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dy centred finite differences
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                     + 1 * ( V[i-1, j+1] * B[i-1, j+1] - V[i-1, j-1] * B[i-1, j-1] ) \
                     + 2 * ( V[i,   j+1] * B[i,   j+1] - V[i,   j-1] * B[i,   j-1] ) \
                     + 1 * ( V[i+1, j+1] * B[i+1, j+1] - V[i+1, j-1] * B[i+1, j-1] ) \
                 ) / self.hy
 
        return result
    
    