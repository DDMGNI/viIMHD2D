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

from PETSc_MHD_Derivatives import  *
from PETSc_MHD_Derivatives cimport *



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
        
        # create local vectors
        self.localB  = da.createLocalVec()
        self.localX  = da.createLocalVec()
        self.localXh = da.createLocalVec()
        
        # create temporary numpy array
        (xs, xe), (ys, ye) = self.da.getRanges()
        self.ty = np.empty((xe-xs, ye-ys, 6))
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(da, nx, ny, hx, hy)
        
    
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
        
        y  = self.da.getVecArray(Y)
        x  = self.da.getVecArray(self.localX)
        xh = self.da.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] ty  = self.ty
        cdef np.ndarray[np.float64_t, ndim=3] tx  = x [...]
        cdef np.ndarray[np.float64_t, ndim=3] txh = xh[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
            
                tf[iy, jy, :] = 0.0
                    
        
        y[xs:xe, ys:ye, :] = ty[:,:,:]
        
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        self.da.globalToLocal(self.Xh, self.localXh)
        
        b  = self.da.getVecArray(B)
        xh = self.da.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] tb  = self.ty
        cdef np.ndarray[np.float64_t, ndim=3] txh = xh[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
            
                tb[iy, jy, :] = 0.0
                  
          
        b[xs:xe, ys:ye, :] = tb[:,:,:]
    


