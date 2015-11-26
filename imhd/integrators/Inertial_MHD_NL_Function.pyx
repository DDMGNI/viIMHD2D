'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport SNES, Mat, Vec

from MHD_Derivatives import  MHD_Derivatives
from MHD_Derivatives cimport MHD_Derivatives



cdef class PETScFunction(object):
    '''
    
    '''
    
    def __init__(self, object da1, object da5,
                 int nx, int ny,
                 double ht, double hx, double hy,
                 double mu, double nu, double eta, double de):
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
        
        # friction, viscosity, resistivity, electron skin depth
        self.mu  = mu
        self.nu  = nu
        self.eta = eta
        self.de  = de
        
        # create history vector
        self.Xh = self.da5.createGlobalVec()
        
        # create local vectors
        self.localX  = da5.createLocalVec()
        self.localXh = da5.createLocalVec()
        
        # create derivatives object
        self.derivatives = MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da5.globalToLocal(X,       self.localX)
        self.da5.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[double, ndim=3] x  = self.da5.getVecArray(self.localX) [...]
        cdef np.ndarray[double, ndim=3] xh = self.da5.getVecArray(self.localXh)[...]
        
        cdef np.ndarray[double, ndim=3] y = self.da5.getVecArray(Y)[...]

        cdef np.ndarray[double, ndim=2] Vx  =  x [:,:,0]
        cdef np.ndarray[double, ndim=2] Vy   = x [:,:,1]
        cdef np.ndarray[double, ndim=2] Bx   = x [:,:,2]
        cdef np.ndarray[double, ndim=2] By   = x [:,:,3]
        cdef np.ndarray[double, ndim=2] Bix  = x [:,:,4]
        cdef np.ndarray[double, ndim=2] Biy  = x [:,:,5]
        cdef np.ndarray[double, ndim=2] P    = x [:,:,6]
        
        cdef np.ndarray[double, ndim=2] Vxh  = xh[:,:,0]
        cdef np.ndarray[double, ndim=2] Vyh  = xh[:,:,1]
        cdef np.ndarray[double, ndim=2] Bxh  = xh[:,:,2]
        cdef np.ndarray[double, ndim=2] Byh  = xh[:,:,3]
        cdef np.ndarray[double, ndim=2] Bixh = xh[:,:,4]
        cdef np.ndarray[double, ndim=2] Biyh = xh[:,:,5]
        cdef np.ndarray[double, ndim=2] Ph   = xh[:,:,6]
        
        cdef double[:,:] Vx_ave  = 0.5 * (Vx  + Vxh )
        cdef double[:,:] Vy_ave  = 0.5 * (Vy  + Vyh )
        cdef double[:,:] Bx_ave  = 0.5 * (Bx  + Bxh )
        cdef double[:,:] By_ave  = 0.5 * (By  + Byh )
        cdef double[:,:] Bix_ave = 0.5 * (Bix + Bixh)
        cdef double[:,:] Biy_ave = 0.5 * (Biy + Biyh)
        cdef double[:,:] P_ave   = 0.5 * (P   + Ph  )

        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = self.dt(Vx,  ix, jx) \
                             - self.dt(Vxh, ix, jx) \
                             + self.derivatives.psix(Vx_ave,  Vy_ave,  Vx_ave, Vy_ave, ix, jx) \
                             - self.derivatives.psix(Bix_ave, Biy_ave, Bx_ave, By_ave, ix, jx) \
                             + self.derivatives.divx_sg(P,  ix, jx) \
                             + 0.5 * self.mu * Vx [ix,jx] \
                             + 0.5 * self.mu * Vxh[ix,jx]
                
                # V_y
                y[iy, jy, 1] = self.dt(Vy,  ix, jx) \
                             - self.dt(Vyh, ix, jx) \
                             + self.derivatives.psiy(Vx_ave,  Vy_ave,  Vx_ave, Vy_ave, ix, jx) \
                             - self.derivatives.psiy(Bix_ave, Biy_ave, Bx_ave, By_ave, ix, jx) \
                             + self.derivatives.divy_sg(P,  ix, jx) \
                             + 0.5 * self.mu * Vy [ix,jx] \
                             + 0.5 * self.mu * Vyh[ix,jx]
                
                # B_x
                y[iy, jy, 2] = Bix[ix, jx] \
                             - self.derivatives.Bix(Bx, By, ix, jx, self.de)
                
                # B_y
                y[iy, jy, 3] = Biy[ix, jx] \
                             - self.derivatives.Biy(Bx, By, ix, jx, self.de)
                
                # Bi_x
                y[iy, jy, 4] = self.dt(Bix,  ix, jx) \
                             - self.dt(Bixh, ix, jx) \
                             + self.derivatives.phix(Vx_ave,  Biy_ave, ix, jx) \
                             - self.derivatives.phix(Bix_ave, Vy_ave,  ix, jx)
                
                # Bi_y
                y[iy, jy, 5] = self.dt(Biy,  ix, jx) \
                             - self.dt(Biyh, ix, jx) \
                             + self.derivatives.phiy(Vx_ave,  Biy_ave, ix, jx) \
                             - self.derivatives.phiy(Bix_ave, Vy_ave,  ix, jx)
                
                # P
                y[iy, jy, 6] = self.derivatives.gradx_simple(Vx_ave, ix, jx) \
                             + self.derivatives.grady_simple(Vy_ave, ix, jx)
                


    @cython.boundscheck(False)
    def timestep(self, np.ndarray[double, ndim=3] x,
                       np.ndarray[double, ndim=3] y):
        
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double[:,:] Vx  = x[:,:,0]
        cdef double[:,:] Vy  = x[:,:,1]
        cdef double[:,:] Bx  = x[:,:,2]
        cdef double[:,:] By  = x[:,:,3]
        cdef double[:,:] Bix = x[:,:,4]
        cdef double[:,:] Biy = x[:,:,5]
        cdef double[:,:] P   = x[:,:,6]

        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = \
                             + self.derivatives.psix(Vx,  Vy,  Vx, Vy, ix, jx) \
                             - self.derivatives.psix(Bix, Biy, Bx, By, ix, jx) \
                             + self.derivatives.divx_sg(P, ix, jx)
                
                # V_y
                y[iy, jy, 1] = \
                             + self.derivatives.psiy(Vx,  Vy,  Vx, Vy, ix, jx) \
                             - self.derivatives.psiy(Bix, Biy, Bx, By, ix, jx) \
                             + self.derivatives.divy_sg(P, ix, jx)
                              
                # B_x
                y[iy, jy, 2] = Bx[ix,jx]
                
                # B_y
                y[iy, jy, 3] = By[ix,jx]
                
                # Bi_x
                y[iy, jy, 4] = \
                             + self.derivatives.phix(Vx, Biy, ix, jx) \
                             - self.derivatives.phix(Bix, Vy, ix, jx)
                    
                # Bi_y
                y[iy, jy, 5] = \
                             + self.derivatives.phiy(Vx, Biy, ix, jx) \
                             - self.derivatives.phiy(Bix, Vy, ix, jx)
                
                # P
                y[iy, jy, 6] = P[ix,jx]
                             
        


    @cython.boundscheck(False)
    cdef double dt(self, double[:,:] x, int i, int j):
        '''
        Time Derivative
        '''
        
        return x[i, j] * self.ht_inv
    
    
    
    @cython.boundscheck(False)
    cdef double dt_x(self, double[:,:] x, int i, int j):
        '''
        Time Derivative
        '''
        
        cdef double result
        
        result = ( \
                     + 1. * x[i,   j-1] \
                     + 2. * x[i,   j  ] \
                     + 1. * x[i,   j+1] \
                 ) * 0.25 * self.ht_inv
                 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef double dt_y(self, double[:,:] x, int i, int j):
        '''
        Time Derivative
        '''
        
        cdef double result
        
        result = ( \
                     + 1. * x[i-1, j  ] \
                     + 2. * x[i,   j  ] \
                     + 1. * x[i+1, j  ] \
                 ) * 0.25 * self.ht_inv
        
        return result
