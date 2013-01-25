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



cdef class PETScPoisson(object):
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
        
        # create auxiliary vectors
        self.dPdx = da1.createGlobalVec()
        self.dPdy = da1.createGlobalVec()
        self.localdPdx = da1.createLocalVec()
        self.localdPdy = da1.createLocalVec()
        
        # create local vectors
        self.localX = da1.createLocalVec()
        self.localR = da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da1.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] dP  = self.da1.getVecArray(self.localX)[...]
        cdef np.ndarray[np.float64_t, ndim=2] tPx = self.da1.getVecArray(self.dPdx)[...]
        cdef np.ndarray[np.float64_t, ndim=2] tPy = self.da1.getVecArray(self.dPdy)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                tPx[iy, jy] = self.derivatives.gradx(dP, ix, jx)
                tPy[iy, jy] = self.derivatives.grady(dP, ix, jx)
        
        
        self.da1.globalToLocal(self.dPdx, self.localdPdx)
        self.da1.globalToLocal(self.dPdy, self.localdPdy)
        
        cdef np.ndarray[np.float64_t, ndim=2] dPdx = self.da1.getVecArray(self.localdPdx)[...]
        cdef np.ndarray[np.float64_t, ndim=2] dPdy = self.da1.getVecArray(self.localdPdy)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                y[iy, jy] = 4. * self.ht * self.derivatives.gradx(dPdx, ix, jx) \
                          + 4. * self.ht * self.derivatives.grady(dPdy, ix, jx)
        

#    @cython.boundscheck(False)
    def formRHS(self, Vec B, Vec R):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da4.globalToLocal(R, self.localR)
        
        r = self.da4.getVecArray(self.localR)
        
        cdef np.ndarray[np.float64_t, ndim=2] b = self.da1.getVecArray(B)[...]
        
#        cdef np.ndarray[np.float64_t, ndim=2] RBx = r[...][:,:,0]
#        cdef np.ndarray[np.float64_t, ndim=2] RBy = r[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] RVx = r[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] RVy = r[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] RP  = r[...][:,:,4]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                b[iy, jy] = RP[ix,jx] \
                          - 4. * self.ht * self.derivatives.gradx(RVx, ix, jx) \
                          - 4. * self.ht * self.derivatives.grady(RVy, ix, jx)


