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
                 np.float64_t ht, np.float64_t hx, np.float64_t hy,
                 eps=0.):
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
        
        # parameter
        self.eps = eps
        
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
        

        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = self.derivatives.dt(Vx,  ix, jx) \
                             - self.derivatives.dt(Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vx,  Vx,  ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vxh, Vx,  ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vx,  Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vxh, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vy,  Vx,  ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vyh, Vx,  ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vy,  Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vyh, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bx,  Bx,  ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bxh, Bx,  ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bx,  Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bxh, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(By,  Bx,  ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Byh, Bx,  ix, jx) \
                             - 0.25 * self.derivatives.fdudy(By,  Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Byh, Bxh, ix, jx) \
                             + 0.5  * self.derivatives.gradx(P,  ix, jx) \
                             + 0.5  * self.derivatives.gradx(Ph, ix, jx)
#                             + 0.5  * self.derivatives.gradx_simple(P,  ix, jx) \
#                             + 0.5  * self.derivatives.gradx_simple(Ph, ix, jx)
#                             + 1.0  * self.derivatives.divx(P,  ix, jx)
#                             + 0.5  * self.derivatives.divx(P,  ix, jx) \
#                             + 0.5  * self.derivatives.divx(Ph, ix, jx)
                              
                # V_y
                y[iy, jy, 1] = self.derivatives.dt(Vy,  ix, jx) \
                             - self.derivatives.dt(Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vx,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vxh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vx,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vxh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vy,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vyh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vy,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vyh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bx,  By,  ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bxh, By,  ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bx,  Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bxh, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(By,  By,  ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Byh, By,  ix, jx) \
                             - 0.25 * self.derivatives.fdudy(By,  Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Byh, Byh, ix, jx) \
                             + 0.5  * self.derivatives.grady(P,  ix, jx) \
                             + 0.5  * self.derivatives.grady(Ph, ix, jx)
#                             + 0.5  * self.derivatives.grady_simple(P,  ix, jx) \
#                             + 0.5  * self.derivatives.grady_simple(Ph, ix, jx)
#                             + 1.0  * self.derivatives.divy(P,  ix, jx)
#                             + 0.5  * self.derivatives.divy(P,  ix, jx) \
#                             + 0.5  * self.derivatives.divy(Ph, ix, jx)
                              
                # B_x
                y[iy, jy, 2] = self.derivatives.dt(Bx,  ix, jx) \
                             - self.derivatives.dt(Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vx,  Bx,  ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vxh, Bx,  ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vx,  Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vxh, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vy,  Bx,  ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vyh, Bx,  ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vy,  Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vyh, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bx,  Vx,  ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bxh, Vx,  ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bx,  Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bxh, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(By,  Vx,  ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Byh, Vx,  ix, jx) \
                             - 0.25 * self.derivatives.fdudy(By,  Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Byh, Vxh, ix, jx)
                    
                # B_y
                y[iy, jy, 3] = self.derivatives.dt(By,  ix, jx) \
                             - self.derivatives.dt(Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vx,  By,  ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vxh, By,  ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vx,  Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Vxh, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vy,  By,  ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vyh, By,  ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vy,  Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Vyh, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bx,  Vy,  ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bxh, Vy,  ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bx,  Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Bxh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(By,  Vy,  ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Byh, Vy,  ix, jx) \
                             - 0.25 * self.derivatives.fdudy(By,  Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Byh, Vyh, ix, jx)
                
                # P
#                y[iy, jy, 4] = self.derivatives.gradx(Vx,  ix, jx) \
#                             + self.derivatives.grady(Vy,  ix, jx) \
#                             - self.eps * self.derivatives.laplace(P, ix, jx)
#                             
#                y[iy, jy, 4] = self.derivatives.divx(Vx,  ix, jx) \
#                             + self.derivatives.divy(Vy,  ix, jx) \
#                             - self.eps * self.derivatives.laplace_fd(P, ix, jx)
#                             
#                y[iy, jy, 4] = self.derivatives.gradx(Vx,  ix, jx) \
#                             + self.derivatives.grady(Vy,  ix, jx)
                             
                y[iy, jy, 4] = self.derivatives.divx_sg(Vx,  ix, jx) \
                             + self.derivatives.divy_sg(Vy,  ix, jx)
                             
#                y[iy, jy, 4] = 0.5 * self.derivatives.gradx(Vx,  ix, jx) \
#                             + 0.5 * self.derivatives.gradx(Vxh, ix, jx) \
#                             + 0.5 * self.derivatives.grady(Vy,  ix, jx) \
#                             + 0.5 * self.derivatives.grady(Vyh, ix, jx)
        
