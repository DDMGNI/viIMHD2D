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

from imhd.integrators.MHD_Derivatives import  MHD_Derivatives
from imhd.integrators.MHD_Derivatives cimport MHD_Derivatives



cdef class PETScSolverPoisson(object):
    '''
    
    '''
    
    def __init__(self, object da2, object da7,
                 int nx, int ny,
                 double ht, double hx, double hy,
                 double mu, double nu, double eta, double de):
        '''
        Constructor
        '''
        
        # distributed array
        self.da2 = da2
        self.da7 = da7
        
        # grid size
        self.nx = nx
        self.ny = ny
        
        # step size
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        # friction, viscosity, resistivity, electron skin depth
        self.mu  = mu
        self.nu  = nu
        self.eta = eta
        self.de  = de
        
        # create history vectors
        self.Bp = da2.createGlobalVec()
        self.Xh = da7.createGlobalVec()
        self.Xp = da7.createGlobalVec()
        
        # create local vectors
        self.localB  = da2.createLocalVec()
        self.localX  = da7.createLocalVec()
        self.localXh = da7.createLocalVec()
        self.localXp = da7.createLocalVec()
        
        # create derivatives object
        self.derivatives = MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        

    def update_previous(self, Vec X):
        X.copy(self.Xp)
    
    
    def update_previous_B(self, Vec B):
        B.copy(self.Bp)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        # copy solution from B to x vector
        x_arr = self.da7.getVecArray(self.Xp)[...]
        b_arr = self.da2.getVecArray(self.Bp)[...]
        
        x_arr[xs:xe, ys:ye, 2] = b_arr[xs:xe, ys:ye, 0]
        x_arr[xs:xe, ys:ye, 3] = b_arr[xs:xe, ys:ye, 1]
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec B, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        self.da7.globalToLocal(self.Xp, self.localX)

        cdef np.ndarray[double, ndim=3] x = self.da7.getVecArray(self.localX)[...]
        
        self.da2.globalToLocal(B, self.localB)
        
        cdef np.ndarray[double, ndim=3] b = self.da2.getVecArray(self.localB)[...]
        cdef np.ndarray[double, ndim=3] y = self.da2.getVecArray(Y)[...]

        cdef double[:,:] Bx  = b[:,:,0]
        cdef double[:,:] By  = b[:,:,1]
        
        cdef double[:,:] Bix = x[:,:,4]
        cdef double[:,:] Biy = x[:,:,5]
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # B_x
                y[iy, jy, 0] = Bix[ix, jx] \
                             - self.derivatives.Bix(Bx, By, ix, jx, self.de)
                
                # B_y
                y[iy, jy, 1] = Biy[ix, jx] \
                             - self.derivatives.Biy(Bx, By, ix, jx, self.de)
                


    @cython.boundscheck(False)
    def formMat(self, Mat A, Mat P = None):
        cdef int i, j, ia, ja, ix, jx
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        A.zeroEntries()
        
        if P != None:
            P.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            
            for j in range(ys, ye):
                jx = j-ys+2
                
                # B_x
                row.index = (i,j)
                row.field = 0
                
                # - dBx + de**2 * dBx_yy
                col.field = 0
                for index, value in [
                        ((i,   j-1), + 1. * self.de**2 / self.hy**2 ),
                        ((i,   j  ), - 2. * self.de**2 / self.hy**2 - 1. ),
                        ((i,   j+1), + 1. * self.de**2 / self.hy**2 ),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                # - de**2 * dBy_xy
                col.field = 1
                for index, value in [
                        ((i-1, j  ), - 1. * self.de**2 / self.hx / self.hy ),
                        ((i,   j  ), + 1. * self.de**2 / self.hx / self.hy ),
                        ((i-1, j+1), + 1. * self.de**2 / self.hx / self.hy ),
                        ((i,   j+1), - 1. * self.de**2 / self.hx / self.hy ),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # B_y
                row.index = (i,j)
                row.field = 1
                
                # - de**2 * dBx_xy
                col.field = 0
                for index, value in [
                        ((i,   j  ), + 1. * self.de**2 / self.hx / self.hy ),
                        ((i,   j-1), - 1. * self.de**2 / self.hx / self.hy ),
                        ((i+1, j  ), - 1. * self.de**2 / self.hx / self.hy ),
                        ((i+1, j-1), + 1. * self.de**2 / self.hx / self.hy ),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                # - dBy + de**2 * dBy_xx
                col.field = 1
                for index, value in [
                        ((i-1, j), + 1. * self.de**2 / self.hx**2 ),
                        ((i,   j), - 2. * self.de**2 / self.hx**2 - 1. ),
                        ((i+1, j), + 1. * self.de**2 / self.hx**2 ),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
        if P != None:
            P.assemble()
        
