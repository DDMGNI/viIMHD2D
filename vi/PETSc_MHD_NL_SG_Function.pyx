'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport DMDA, SNES, Mat, Vec

from PETSc_MHD_Derivatives import  PETSc_MHD_Derivatives
from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScFunction(object):
    '''
    
    '''
    
    def __init__(self, DMDA da1, DMDA da5,
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
        
        self.ht_inv = 1. / ht
        self.hx_inv = 1. / hx
        self.hy_inv = 1. / hy
        
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
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = self.dt_x(Vx,  ix, jx) \
                             - self.dt_x(Vxh, ix, jx) \
                             - 0.25 * self.ave_xt(Vy, Vyh, ix, jx  ) * ( self.curl(Vx, Vy, ix, jx  ) + self.curl(Vxh, Vyh, ix, jx  ) ) \
                             - 0.25 * self.ave_xt(Vy, Vyh, ix, jx-1) * ( self.curl(Vx, Vy, ix, jx-1) + self.curl(Vxh, Vyh, ix, jx-1) ) \
                             + 0.25 * self.ave_xt(By, Byh, ix, jx  ) * ( self.curl(Bx, By, ix, jx  ) + self.curl(Bxh, Byh, ix, jx  ) ) \
                             + 0.25 * self.ave_xt(By, Byh, ix, jx-1) * ( self.curl(Bx, By, ix, jx-1) + self.curl(Bxh, Byh, ix, jx-1) ) \
                             + 0.5  * self.derivatives.gradx_sg(P,  ix, jx) \
                             + 0.5  * self.derivatives.gradx_sg(Ph, ix, jx)
#                             + self.derivatives.gradx_sg(P,  ix, jx)
                
                # V_y
                y[iy, jy, 1] = self.dt_y(Vy,  ix, jx) \
                             - self.dt_y(Vyh, ix, jx) \
                             + 0.25 * self.ave_yt(Vx, Vxh, ix,   jx) * ( self.curl(Vx, Vy, ix,   jx) + self.curl(Vxh, Vyh, ix,   jx) ) \
                             + 0.25 * self.ave_yt(Vx, Vxh, ix-1, jx) * ( self.curl(Vx, Vy, ix-1, jx) + self.curl(Vxh, Vyh, ix-1, jx) ) \
                             - 0.25 * self.ave_yt(Bx, Bxh, ix,   jx) * ( self.curl(Bx, By, ix,   jx) + self.curl(Bxh, Byh, ix,   jx) ) \
                             - 0.25 * self.ave_yt(Bx, Bxh, ix-1, jx) * ( self.curl(Bx, By, ix-1, jx) + self.curl(Bxh, Byh, ix-1, jx) ) \
                             + 0.5  * self.derivatives.grady_sg(P,  ix, jx) \
                             + 0.5  * self.derivatives.grady_sg(Ph, ix, jx)
#                             + self.derivatives.grady_sg(P,  ix, jx)
                
                # B_x
                y[iy, jy, 2] = self.dt_x(Bx,  ix, jx) \
                             - self.dt_x(Bxh, ix, jx) \
                             - self.ave_yt(Vx, Vxh, ix, jx  ) * self.ave_xt(By, Byh, ix, jx  ) * self.hy_inv \
                             + self.ave_yt(Vx, Vxh, ix, jx-1) * self.ave_xt(By, Byh, ix, jx-1) * self.hy_inv \
                             + self.ave_yt(Bx, Bxh, ix, jx  ) * self.ave_xt(Vy, Vyh, ix, jx  ) * self.hy_inv \
                             - self.ave_yt(Bx, Bxh, ix, jx-1) * self.ave_xt(Vy, Vyh, ix, jx-1) * self.hy_inv
                
                # B_y
                y[iy, jy, 3] = self.dt_y(By,  ix, jx) \
                             - self.dt_y(Byh, ix, jx) \
                             + self.ave_yt(Vx, Vxh, ix,   jx) * self.ave_xt(By, Byh, ix,   jx) * self.hx_inv \
                             - self.ave_yt(Vx, Vxh, ix-1, jx) * self.ave_xt(By, Byh, ix-1, jx) * self.hx_inv \
                             - self.ave_yt(Bx, Bxh, ix,   jx) * self.ave_xt(Vy, Vyh, ix,   jx) * self.hx_inv \
                             + self.ave_yt(Bx, Bxh, ix-1, jx) * self.ave_xt(Vy, Vyh, ix-1, jx) * self.hx_inv
                
                # P
                y[iy, jy, 4] = self.derivatives.divx_sg(Vx,  ix, jx) \
                             + self.derivatives.divy_sg(Vy,  ix, jx)
        
#                y[iy, jy, 4] = 0.5 * self.derivatives.divx_sg(Vx,  ix, jx) \
#                             + 0.5 * self.derivatives.divx_sg(Vxh, ix, jx) \
#                             + 0.5 * self.derivatives.divy_sg(Vy,  ix, jx) \
#                             + 0.5 * self.derivatives.divy_sg(Vyh, ix, jx)
        

    @cython.boundscheck(False)
    cdef np.float64_t dt_x(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative of x component
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * x[i,   j-1] \
                   + 2. * x[i,   j  ] \
                   + 1. * x[i,   j+1] \
                 ) * 0.25 * self.ht_inv
        
#        result = ( \
#                   + x[i,   j  ] \
#                   + x[i,   j+1] \
#                 ) * 0.5 * self.ht_inv
        
        return result


    @cython.boundscheck(False)
    cdef np.float64_t dt_y(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative of y component
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * x[i-1, j  ] \
                   + 2. * x[i,   j  ] \
                   + 1. * x[i+1, j  ] \
                 ) * 0.25 * self.ht_inv
        
#        result = ( \
#                   + x[i,   j  ] \
#                   + x[i+1, j  ] \
#                 ) * 0.5 * self.ht_inv
        
        return result


    @cython.boundscheck(False)
    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_x
        '''
        
        return ( x[i+1, j] - x[i, j] ) * self.hx_inv
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_y
        '''
        
        return ( x[i, j+1] - x[i, j] ) * self.hy_inv
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t curl(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.ndarray[np.float64_t, ndim=2] y,
                                 np.uint64_t i, np.uint64_t j):
        '''
        2D curl
        '''
        
        return self.dx(y, i, j) - self.dy(x, i, j)
        

    @cython.boundscheck(False)
    cdef np.float64_t ave_xt(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] xh,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Average in x and t
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                   + x [i,   j] \
                   + x [i+1, j] \
                   + xh[i,   j] \
                   + xh[i+1, j] \
                 )
        
        return result


    @cython.boundscheck(False)
    cdef np.float64_t ave_yt(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] xh,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Average in y and t
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                   + x [i, j  ] \
                   + x [i, j+1] \
                   + xh[i, j  ] \
                   + xh[i, j+1] \
                 )
        
        return result


