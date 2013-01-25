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
        

        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = self.dt(Vx,  ix, jx) \
                             - self.dt(Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_ux(Vx,  Vx,  ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_ux(Vxh, Vx,  ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_ux(Vx,  Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_ux(Vxh, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_ux(Vy,  Vx,  ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_ux(Vyh, Vx,  ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_ux(Vy,  Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_ux(Vyh, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_ux(Bx,  Bx,  ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_ux(Bxh, Bx,  ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_ux(Bx,  Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_ux(Bxh, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_ux(By,  Bx,  ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_ux(Byh, Bx,  ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_ux(By,  Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_ux(Byh, Bxh, ix, jx) \
                             + 1.0  * self.derivatives.gradx_fv(P,  ix, jx)
#                             + 0.5  * self.derivatives.gradx_fv(P,  ix, jx) \
#                             + 0.5  * self.derivatives.gradx_fv(Ph, ix, jx)
                              
                # V_y
                y[iy, jy, 1] = self.dt(Vy,  ix, jx) \
                             - self.dt(Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_uy(Vx,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_uy(Vxh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_uy(Vx,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_uy(Vxh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_uy(Vy,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_uy(Vyh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_uy(Vy,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_uy(Vyh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_uy(Bx,  By,  ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_uy(Bxh, By,  ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_uy(Bx,  Byh, ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_uy(Bxh, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_uy(By,  By,  ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_uy(Byh, By,  ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_uy(By,  Byh, ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_uy(Byh, Byh, ix, jx) \
                             + 1.0  * self.derivatives.grady_fv(P,  ix, jx)
#                             + 0.5  * self.derivatives.grady_fv(P,  ix, jx) \
#                             + 0.5  * self.derivatives.grady_fv(Ph, ix, jx)
                              
                # B_x
                y[iy, jy, 2] = self.dt(Bx,  ix, jx) \
                             - self.dt(Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_ux(Vx,  Bx,  ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_ux(Vxh, Bx,  ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_ux(Vx,  Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_ux(Vxh, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_ux(Vy,  Bx,  ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_ux(Vyh, Bx,  ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_ux(Vy,  Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_ux(Vyh, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_ux(Bx,  Vx,  ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_ux(Bxh, Vx,  ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_ux(Bx,  Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_ux(Bxh, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_ux(By,  Vx,  ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_ux(Byh, Vx,  ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_ux(By,  Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_ux(Byh, Vxh, ix, jx)
                    
                # B_y
                y[iy, jy, 3] = self.dt(By,  ix, jx) \
                             - self.dt(Byh, ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_uy(Vx,  By,  ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_uy(Vxh, By,  ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_uy(Vx,  Byh, ix, jx) \
                             + 0.25 * self.derivatives.fx_dx_uy(Vxh, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_uy(Vy,  By,  ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_uy(Vyh, By,  ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_uy(Vy,  Byh, ix, jx) \
                             + 0.25 * self.derivatives.fy_dy_uy(Vyh, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_uy(Bx,  Vy,  ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_uy(Bxh, Vy,  ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_uy(Bx,  Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fx_dx_uy(Bxh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_uy(By,  Vy,  ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_uy(Byh, Vy,  ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_uy(By,  Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fy_dy_uy(Byh, Vyh, ix, jx)
                
                # P
                y[iy, jy, 4] = self.derivatives.divx_sg(Vx, ix, jx) \
                             + self.derivatives.divy_sg(Vy, ix, jx)
                             
#                y[iy, jy, 4] = 0.5 * self.derivatives.divx_sg(Vx,  ix, jx) \
#                             + 0.5 * self.derivatives.divx_sg(Vxh, ix, jx) \
#                             + 0.5 * self.derivatives.divy_sg(Vy,  ix, jx) \
#                             + 0.5 * self.derivatives.divy_sg(Vyh, ix, jx)
        


    @cython.boundscheck(False)
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1.  * x[i-2, j  ] \
                     + 2.  * x[i-1, j-1] \
                     + 8.  * x[i-1, j  ] \
                     + 2.  * x[i-1, j+1] \
                     + 1.  * x[i,   j-2] \
                     + 8.  * x[i,   j-1] \
                     + 20. * x[i,   j  ] \
                     + 8.  * x[i,   j+1] \
                     + 1.  * x[i,   j+2] \
                     + 2.  * x[i+1, j-1] \
                     + 8.  * x[i+1, j  ] \
                     + 2.  * x[i+1, j+1] \
                     + 1.  * x[i+2, j  ] \
                 ) * self.ht_inv / 64.
        
        
        return result



