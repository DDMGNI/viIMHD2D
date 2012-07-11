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
    
    def __init__(self, DA da,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        assert da.getDim() == 2
        
        # distributed array
        self.da = da
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        
        # create history vector
        self.Xh = self.da.createGlobalVec()
        self.P  = self.da.createGlobalVec()
        self.V  = self.da.createGlobalVec()
        
        # create local vectors
        self.localB  = self.da.createLocalVec()
        self.localX  = self.da.createLocalVec()
        self.localXh = self.da.createLocalVec()
        self.localP  = self.da.createLocalVec()
        self.localV  = self.da.createLocalVec()
        
        # create global RK4 vectors
        self.X1 = self.da.createGlobalVec()
        self.X2 = self.da.createGlobalVec()
        self.X3 = self.da.createGlobalVec()
        self.X4 = self.da.createGlobalVec()
        
        # create local RK4 vectors
        self.localX1 = self.da.createLocalVec()
        self.localX2 = self.da.createLocalVec()
        self.localX3 = self.da.createLocalVec()
        self.localX4 = self.da.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(da, nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        x  = self.da.getVecArray(X)
        xh = self.da.getVecArray(self.Xh)
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        xh[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        self.da.globalToLocal(X,       self.localX)
        self.da.globalToLocal(self.Xh, self.localXh)
        
        x  = self.da.getVecArray(self.localX)
        xh = self.da.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] ty  = self.da.getVecArray(Y)[...]

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
        
        
#        cdef np.ndarray[np.float64_t, ndim=2] p = self.da.getVecArray(self.P)[...][:,:,0]
#        
#        for i in np.arange(xs, xe):
#            ix = i-xs+2
#            iy = i-xs
#            
#            for j in np.arange(ys, ye):
#                jx = j-ys+2
#                jy = j-ys
#                
#                p[iy,jy] = 0.5 * (Vx[ix,jx]**2 + Vy[ix,jx]**2)
#                
#        self.da.globalToLocal(self.P, self.localP)
#        
#        p = self.da.getVecArray(self.localP)[...][:,:,0]
        
        
        v = self.da.getVecArray(self.V)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] tVx = v[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] tVy = v[:,:,1]
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                tVx[iy, jy] = self.psi_x(Vx,  Vy, ix, jx) \
                            - self.psi_x(Bx,  By, ix, jx)
                
                tVy[iy, jy] = self.psi_y(Vx,  Vy, ix, jx) \
                            - self.psi_y(Bx,  By, ix, jx)
                
#                tVx[iy, jy] = 0.5 * (self.psi_x(Vx, Vy, ix, jx) + self.psi_x(Vxh, Vyh, ix, jx)) \
#                            - 0.5 * (self.psi_x(Bx, By, ix, jx) + self.psi_x(Bxh, Byh, ix, jx))
#                
#                tVy[iy, jy] = 0.5 * (self.psi_y(Vx, Vy, ix, jx) + self.psi_y(Vxh, Vyh, ix, jx)) \
#                            - 0.5 * (self.psi_y(Bx, By, ix, jx) + self.psi_y(Bxh, Byh, ix, jx))
                
        self.da.globalToLocal(self.V, self.localV)
        
        v = self.da.getVecArray(self.localV)[...]
        
        tVx = v[:,:,0]
        tVy = v[:,:,1]
         
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # B_x
                ty[iy, jy, 0] = self.dt_yave(Bx, ix, jx) \
                              + 0.5 * self.phi_x(Vxh,  Vyh, 0.5*(Bx+Bxh), 0.5*(By+Byh), ix, jx)
                    
                # B_y
                ty[iy, jy, 1] = self.dt_xave(By, ix, jx) \
                              + 0.5 * self.phi_y(Vxh,  Vyh, 0.5*(Bx+Bxh), 0.5*(By+Byh), ix, jx)
                
                # V_x
                ty[iy, jy, 2] = self.dt_yave(Vx, ix, jx) \
                              + 0.5 * self.psi_x(Vx,  Vy, ix, jx) \
                              - 0.5 * self.psi_x(Bx,  By, ix, jx) \
                              + 1.0 * self.derivatives.gradx(P, ix, jx)
                    
                # V_y
                ty[iy, jy, 3] = self.dt_xave(Vy, ix, jx) \
                              + 0.5 * self.psi_y(Vx,  Vy, ix, jx) \
                              - 0.5 * self.psi_y(Bx,  By, ix, jx) \
                              + 1.0 * self.derivatives.grady(P, ix, jx)
                    
                # P
                ty[iy, jy, 4] = self.laplace(P, ix, jx) \
                              + self.dx1(tVx, ix, jx) \
                              + self.dy1(tVy, ix, jx)
        
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        self.da.globalToLocal(self.Xh, self.localXh)
        
        xh = self.da.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] tb  = self.da.getVecArray(B)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,3]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                tb[iy, jy, 0] = self.dt_yave(Bxh, ix, jx)
                
                tb[iy, jy, 1] = self.dt_xave(Byh, ix, jx)
                
                tb[iy, jy, 2] = self.dt_yave(Vxh, ix, jx) \
                              - 0.5 * self.psi_x(Vxh,  Vyh, ix, jx) \
                              + 0.5 * self.psi_x(Bxh,  Byh, ix, jx)
                
                tb[iy, jy, 3] = self.dt_xave(Vyh, ix, jx) \
                              - 0.5 * self.psi_y(Vxh,  Vyh, ix, jx) \
                              + 0.5 * self.psi_y(Bxh,  Byh, ix, jx)
                
                tb[iy, jy, 4] = 0.0


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
        
        tx[:,:,:] = tx + self.ht * (tx1 + 2.*tx2 + 2.*tx3 + tx4) / 6.

    
    
#    @cython.boundscheck(False)
    cdef timestep(self, np.ndarray[np.float64_t, ndim=3] tx,
                        np.ndarray[np.float64_t, ndim=3] ty):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = tx[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = tx[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = tx[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = tx[:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] Ph  = tx[:,:,4]
         
         
#        cdef np.ndarray[np.float64_t, ndim=2] p = self.da.getVecArray(self.P)[...][:,:,0]
#        
#        for i in np.arange(xs, xe):
#            ix = i-xs+2
#            iy = i-xs
#            
#            for j in np.arange(ys, ye):
#                jx = j-ys+2
#                jy = j-ys
#                
#                p[iy,jy] = 0.5 * (Vxh[ix,jx]**2 + Vyh[ix,jx]**2)
#                
#        self.da.globalToLocal(self.P, self.localP)
#        
#        p = self.da.getVecArray(self.localP)[...][:,:,0]
        
        
        for j in np.arange(ys, ye):
            jx = j-ys+2
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+2
                iy = i-xs

                # B_x
                ty[iy, jy, 0] = \
                              - self.phi_x(Vxh, Vyh, Bxh, Byh, ix, jx)
                
                # B_y
                ty[iy, jy, 1] = \
                              - self.phi_y(Vxh, Vyh, Bxh, Byh, ix, jx)
                
                # V_x
                ty[iy, jy, 2] = \
                              - self.psi_x(Vxh, Vyh, ix, jx) \
                              + self.psi_x(Bxh, Byh, ix, jx) \
                              - self.derivatives.gradx(Ph, ix, jx)
                
                # V_y
                ty[iy, jy, 3] = \
                              - self.psi_y(Vxh, Vyh, ix, jx) \
                              + self.psi_y(Bxh, Byh, ix, jx) \
                              - self.derivatives.gradx(Ph, ix, jx)

                ty[iy, jy, 4] = 0.0
        
        
        
    cdef np.float64_t psi_x(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t result
        
        result = 1. * self.omega(Vx, Vy, i, j-1) * self.ave_x(Vy, i, j-1) \
               + 2. * self.omega(Vx, Vy, i, j  ) * self.ave_x(Vy, i, j  ) \
               + 1. * self.omega(Vx, Vy, i, j+1) * self.ave_x(Vy, i, j+1)
        
        return - 0.25 * result
        
        
    cdef np.float64_t psi_y(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t result
        
        result = 1. * self.omega(Vx, Vy, i-1, j) * self.ave_y(Vx, i-1, j) \
               + 2. * self.omega(Vx, Vy, i,   j) * self.ave_y(Vx, i,   j) \
               + 1. * self.omega(Vx, Vy, i+1, j) * self.ave_y(Vx, i+1, j)
        
        return + 0.25 * result
        
        
    cdef np.float64_t phi_x(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.ndarray[np.float64_t, ndim=2] Bx,
                                  np.ndarray[np.float64_t, ndim=2] By,
                                  np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t result
        
        result = Vx[i,j+1] * By[i,j+1] - Vx[i,j-1] * By[i,j-1] \
               - Vy[i,j+1] * Bx[i,j+1] + Vy[i,j-1] * Bx[i,j-1]
        
        return - 0.5 * result * self.hy
        
        
    cdef np.float64_t phi_y(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.ndarray[np.float64_t, ndim=2] Bx,
                                  np.ndarray[np.float64_t, ndim=2] By,
                                  np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t result
        
        result = Vx[i+1,j] * By[i+1,j] - Vx[i-1,j] * By[i-1,j] \
               - Vy[i+1,j] * Bx[i+1,j] + Vy[i-1,j] * Bx[i-1,j]
        
        return + 0.5 * result * self.hx
        
        
    cdef np.float64_t omega(self, np.ndarray[np.float64_t, ndim=2] Vx,
                                  np.ndarray[np.float64_t, ndim=2] Vy,
                                  np.uint64_t i, np.uint64_t j):
        
        return (Vy[i+1,j] - Vy[i-1,j] ) / (2.*self.hx) \
             - (Vx[i,j+1] - Vx[i,j-1] ) / (2.*self.hy) 
    
    
    cdef np.float64_t ave_x(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        
        return 0.25 * ( x[i-1,j] + 2.*x[i,j] + x[i+1,j] )
    
    
    cdef np.float64_t ave_y(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        
        return 0.25 * ( x[i,j-1] + 2.*x[i,j] + x[i,j+1] )
        

#    @cython.boundscheck(False)
    cdef np.float64_t dt_xave(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        return self.ave_x(x, i, j) / self.ht
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dt_yave(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        return self.ave_y(x, i, j) / self.ht
    
    

#    @cython.boundscheck(False)
    cdef np.float64_t laplace(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: Laplace operator (averaged)
        '''
        
        cdef np.float64_t result
        
#        result = ( \
#                   + 1. * x[i-1, j  ] \
#                   - 2. * x[i,   j  ] \
#                   + 1. * x[i+1, j  ] \
#                 ) / self.hx**2 \
#               + ( \
#                   + 1. * x[i,   j-1] \
#                   - 2. * x[i,   j  ] \
#                   + 1. * x[i,   j+1] \
#                 ) / self.hy**2
        
        result = 0.25 * ( \
                 ( \
                   + 1. * x[i-1, j-1] \
                   - 2. * x[i,   j-1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i-1, j  ] \
                   - 4. * x[i,   j  ] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i-1, j+1] \
                   - 2. * x[i,   j+1] \
                   + 1. * x[i+1, j+1] \
                 ) / self.hx**2 \
               + ( \
                   + 1. * x[i-1, j-1] \
                   - 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   - 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   - 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) / self.hy**2 \
               )
 
        return result
    

#    @cython.boundscheck(False)
    cdef np.float64_t dx1(self, np.ndarray[np.float64_t, ndim=2] x,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dx centred finite differences
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                     + 1 * ( x[i+1, j-1] - x[i-1, j-1] ) \
                     + 2 * ( x[i+1, j  ] - x[i-1, j  ] ) \
                     + 1 * ( x[i+1, j+1] - x[i-1, j+1] ) \
                 ) / self.hx
 
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dy1(self, np.ndarray[np.float64_t, ndim=2] x,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dy centred finite differences
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                     + 1 * ( x[i-1, j+1] - x[i-1, j-1] ) \
                     + 2 * ( x[i,   j+1] - x[i,   j-1] ) \
                     + 1 * ( x[i+1, j+1] - x[i+1, j-1] ) \
                 ) / self.hy
 
        return result
