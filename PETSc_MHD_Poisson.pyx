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

from PETSc_MHD_Derivatives import  PETSc_MHD_Derivatives
from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives


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
        self.localV = da4.createLocalVec()
        self.localX = da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(da4, nx, ny, ht, hx, hy)
        
    
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
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                y[iy, jy] = self.derivatives.laplace(x, ix, jx)
        
