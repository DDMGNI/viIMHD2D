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
    
    def __init__(self, DA da1, DA da5,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        # distributed array
        self.da1 = da1
        self.da5 = da5
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        self.ht_inv = 1. / self.ht
        
        # create history vector
        self.Xh = self.da5.createGlobalVec()
        
        # create local vectors
        self.localX  = da5.createLocalVec()
        self.localXh = da5.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
            
    @cython.boundscheck(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da5.globalToLocal(X,       self.localX)
        self.da5.globalToLocal(self.Xh, self.localXh)
        
        x  = self.da5.getVecArray(self.localX)
        xh = self.da5.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] y = self.da5.getVecArray(Y)[...]

        cdef np.ndarray[np.float64_t, ndim=2] Vx  = x [...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vy  = x [...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bx  = x [...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] By  = x [...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] P   = x [...][:,:,4]
        
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] Ph  = xh[...][:,:,4]
        
        cdef np.ndarray[np.float64_t, ndim=2] Vx_ave = 0.5 * (Vx + Vxh)
        cdef np.ndarray[np.float64_t, ndim=2] Vy_ave = 0.5 * (Vy + Vyh)
        cdef np.ndarray[np.float64_t, ndim=2] Bx_ave = 0.5 * (Bx + Bxh)
        cdef np.ndarray[np.float64_t, ndim=2] By_ave = 0.5 * (By + Byh)
        cdef np.ndarray[np.float64_t, ndim=2] P_ave  = 0.5 * (P  + Ph )

        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
#                y[iy, jy, 0] = self.dt_x(Vx,  ix, jx) \
#                             - self.dt_x(Vxh, ix, jx) \
                y[iy, jy, 0] = self.dt(Vx,  ix, jx) \
                             - self.dt(Vxh, ix, jx) \
                             + 0.5 * self.derivatives.psix(Vx,  Vy,  ix, jx) \
                             + 0.5 * self.derivatives.psix(Vxh, Vyh, ix, jx) \
                             - 0.5 * self.derivatives.psix(Bx,  By,  ix, jx) \
                             - 0.5 * self.derivatives.psix(Bxh, Byh, ix, jx) \
                             + 1.0 * self.derivatives.divx_sg(P,  ix, jx)
#                             + self.derivatives.gradx_sg(P,  ix, jx)
#                             + 0.5 * self.derivatives.gradx_sg(P,  ix, jx) \
#                             + 0.5 * self.derivatives.gradx_sg(Ph, ix, jx)
#                             - self.derivatives.psix(Vx_ave,  Vy_ave,  ix, jx) \
#                             + self.derivatives.psix(Bx_ave,  By_ave,  ix, jx) \
#                             - 0.25 * self.derivatives.psix(Vx,  Vy,  ix, jx) \
#                             - 0.25 * self.derivatives.psix(Vx,  Vyh, ix, jx) \
#                             - 0.25 * self.derivatives.psix(Vxh, Vy,  ix, jx) \
#                             - 0.25 * self.derivatives.psix(Vxh, Vyh, ix, jx) \
#                             + 0.25 * self.derivatives.psix(Bx,  By,  ix, jx) \
#                             + 0.25 * self.derivatives.psix(Bx,  Byh, ix, jx) \
#                             + 0.25 * self.derivatives.psix(Bxh, By,  ix, jx) \
#                             + 0.25 * self.derivatives.psix(Bxh, Byh, ix, jx) \
                
                # V_y
#                y[iy, jy, 1] = self.dt_y(Vy,  ix, jx) \
#                             - self.dt_y(Vyh, ix, jx) \
                y[iy, jy, 1] = self.dt(Vy,  ix, jx) \
                             - self.dt(Vyh, ix, jx) \
                             + 0.5 * self.derivatives.psiy(Vx,  Vy,  ix, jx) \
                             + 0.5 * self.derivatives.psiy(Vxh, Vyh, ix, jx) \
                             - 0.5 * self.derivatives.psiy(Bx,  By,  ix, jx) \
                             - 0.5 * self.derivatives.psiy(Bxh, Byh, ix, jx) \
                             + 1.0 * self.derivatives.divy_sg(P,  ix, jx)
#                             + self.derivatives.grady_sg(P,  ix, jx)
#                             + 0.5 * self.derivatives.grady_sg(P,  ix, jx) \
#                             + 0.5 * self.derivatives.grady_sg(Ph, ix, jx)
#                             + self.derivatives.psiy(Vx_ave,  Vy_ave,  ix, jx) \
#                             - self.derivatives.psiy(Bx_ave,  By_ave,  ix, jx) \
#                             + 0.25 * self.derivatives.psiy(Vx,  Vy,  ix, jx) \
#                             + 0.25 * self.derivatives.psiy(Vx,  Vyh, ix, jx) \
#                             + 0.25 * self.derivatives.psiy(Vxh, Vy,  ix, jx) \
#                             + 0.25 * self.derivatives.psiy(Vxh, Vyh, ix, jx) \
#                             - 0.25 * self.derivatives.psiy(Bx,  By,  ix, jx) \
#                             - 0.25 * self.derivatives.psiy(Bx,  Byh, ix, jx) \
#                             - 0.25 * self.derivatives.psiy(Bxh, By,  ix, jx) \
#                             - 0.25 * self.derivatives.psiy(Bxh, Byh, ix, jx) \
                              
                # B_x
#                y[iy, jy, 2] = self.dt_x(Bx,  ix, jx) \
#                             - self.dt_x(Bxh, ix, jx) \
                y[iy, jy, 2] = self.dt(Bx,  ix, jx) \
                             - self.dt(Bxh, ix, jx) \
                             + self.derivatives.phix(Vxh,  By_ave,  ix, jx) \
                             - self.derivatives.phix(Bx_ave,  Vyh,  ix, jx) \
#                             - self.derivatives.phix(Vx_ave,  By_ave,  ix, jx) \
#                             + self.derivatives.phix(Bx_ave,  Vy_ave,  ix, jx) \
#                             - 0.25 * self.derivatives.phix(Vx,  By,  ix, jx) \
#                             - 0.25 * self.derivatives.phix(Vx,  Byh, ix, jx) \
#                             - 0.25 * self.derivatives.phix(Vxh, By,  ix, jx) \
#                             - 0.25 * self.derivatives.phix(Vxh, Byh, ix, jx) \
#                             + 0.25 * self.derivatives.phix(Bx,  Vy,  ix, jx) \
#                             + 0.25 * self.derivatives.phix(Bx,  Vyh, ix, jx) \
#                             + 0.25 * self.derivatives.phix(Bxh, Vy,  ix, jx) \
#                             + 0.25 * self.derivatives.phix(Bxh, Vyh, ix, jx)
                    
                # B_y
#                y[iy, jy, 3] = self.dt_y(By,  ix, jx) \
#                             - self.dt_y(Byh, ix, jx) \
                y[iy, jy, 3] = self.dt(By,  ix, jx) \
                             - self.dt(Byh, ix, jx) \
                             + self.derivatives.phiy(Vxh,  By_ave,  ix, jx) \
                             - self.derivatives.phiy(Bx_ave,  Vyh,  ix, jx) \
#                             + self.derivatives.phiy(Vx_ave,  By_ave,  ix, jx) \
#                             - self.derivatives.phiy(Bx_ave,  Vy_ave,  ix, jx) \
#                             + 0.25 * self.derivatives.phiy(Vx,  By,  ix, jx) \
#                             + 0.25 * self.derivatives.phiy(Vx,  Byh, ix, jx) \
#                             + 0.25 * self.derivatives.phiy(Vxh, By,  ix, jx) \
#                             + 0.25 * self.derivatives.phiy(Vxh, Byh, ix, jx) \
#                             - 0.25 * self.derivatives.phiy(Bx,  Vy,  ix, jx) \
#                             - 0.25 * self.derivatives.phiy(Bx,  Vyh, ix, jx) \
#                             - 0.25 * self.derivatives.phiy(Bxh, Vy,  ix, jx) \
#                             - 0.25 * self.derivatives.phiy(Bxh, Vyh, ix, jx)
                
                # P
                y[iy, jy, 4] = self.derivatives.gradx_simple(Vx, ix, jx) \
                             + self.derivatives.grady_simple(Vy, ix, jx)
                             
#                y[iy, jy, 4] = self.derivatives.divx_sg(Vx, ix, jx) \
#                             + self.derivatives.divy_sg(Vy, ix, jx)
                             
#                y[iy, jy, 4] = 0.5 * self.derivatives.divx_sg(Vx,  ix, jx) \
#                             + 0.5 * self.derivatives.divx_sg(Vxh, ix, jx) \
#                             + 0.5 * self.derivatives.divy_sg(Vy,  ix, jx) \
#                             + 0.5 * self.derivatives.divy_sg(Vyh, ix, jx)
        


    @cython.boundscheck(False)
    def timestep(self, np.ndarray[np.float64_t, ndim=3] x,
                       np.ndarray[np.float64_t, ndim=3] y):
        
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] By = x[:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] P  = x[:,:,4]

        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = \
                             + self.derivatives.psix(Vx, Vy, ix, jx) \
                             - self.derivatives.psix(Bx, By, ix, jx) \
                             + self.derivatives.divx_sg(P, ix, jx)
                
                # V_y
                y[iy, jy, 1] = \
                             + self.derivatives.psiy(Vx, Vy, ix, jx) \
                             - self.derivatives.psiy(Bx, By, ix, jx) \
                             + self.derivatives.divy_sg(P, ix, jx)
                              
                # B_x
                y[iy, jy, 2] = \
                             + self.derivatives.phix(Vx, By, ix, jx) \
                             - self.derivatives.phix(Bx, Vy, ix, jx)
                    
                # B_y
                y[iy, jy, 3] = \
                             + self.derivatives.phiy(Vx, By, ix, jx) \
                             - self.derivatives.phiy(Bx, Vy, ix, jx)
                
                # P
                y[iy, jy, 4] = P[ix,jx]
                             
        


    @cython.boundscheck(False)
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        return x[i, j] * self.ht_inv
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t dt_x(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * x[i,   j-1] \
                     + 2. * x[i,   j  ] \
                     + 1. * x[i,   j+1] \
                 ) * 0.25 * self.ht_inv
                 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t dt_y(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * x[i-1, j  ] \
                     + 2. * x[i,   j  ] \
                     + 1. * x[i+1, j  ] \
                 ) * 0.25 * self.ht_inv
        
        return result
