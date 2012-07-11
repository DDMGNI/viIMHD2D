'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, Mat, Vec


cdef class PETScPoissonSolver(object):
    '''
    
    '''
    
    def __init__(self, DA da1, DA da4, Vec X,
                 np.uint64_t  nx, np.uint64_t  ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        assert da1.getDim() == 2
        assert da4.getDim() == 2
        
        # disstributed array
        self.da1 = da1
        self.da4 = da4
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        # save solution vector
        self.X = X

        self.V = da4.createGlobalVec()
        
        # create local vectors
        self.localP = da1.createLocalVec()
#        self.localU = da4.createLocalVec()
        self.localV = da4.createLocalVec()
        self.localX = da4.createLocalVec()
        
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localP)
        
        cdef np.ndarray[np.float64_t, ndim=2] y = self.da1.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] x = self.da1.getVecArray(self.localP)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                y[iy, jy] = self.laplace(x, ix, jx)
            
        
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da4.globalToLocal(self.X, self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=2] b = self.da1.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=3] x = self.da4.getVecArray(self.localX)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By = x[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[...][:,:,3]
        
        
        v = self.da4.getVecArray(self.V)[...]
        
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
                
        self.da4.globalToLocal(self.V, self.localV)
        
        v = self.da4.getVecArray(self.localV)[...]
        
        tVx = v[:,:,0]
        tVy = v[:,:,1]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                b[iy, jy] = - self.dx1(tVx, ix, jx) \
                            - self.dy1(tVy, ix, jx)

            
        
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
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dx centred finite differences
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1 * ( V[i+1, j-1] * B[i+1, j-1] - V[i-1, j-1] * B[i-1, j-1] ) \
                     + 2 * ( V[i+1, j  ] * B[i+1, j  ] - V[i-1, j  ] * B[i-1, j  ] ) \
                     + 1 * ( V[i+1, j+1] * B[i+1, j+1] - V[i-1, j+1] * B[i-1, j+1] ) \
                 ) * 0.25 / self.hx
 
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dy centred finite differences
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1 * ( V[i-1, j+1] * B[i-1, j+1] - V[i-1, j-1] * B[i-1, j-1] ) \
                     + 2 * ( V[i,   j+1] * B[i,   j+1] - V[i,   j-1] * B[i,   j-1] ) \
                     + 1 * ( V[i+1, j+1] * B[i+1, j+1] - V[i+1, j-1] * B[i+1, j-1] ) \
                 ) * 0.25 / self.hy
 
        return result
    
    
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
        
