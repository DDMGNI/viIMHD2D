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



cdef class PETScSolver(object):
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
        
        # create local vectors
        self.localR  = da4.createLocalVec()
        self.localP  = da1.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
#    @cython.boundscheck(False)
    def solve(self, Vec R, Vec P, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(R, self.localR)
        self.da1.globalToLocal(P, self.localP)
        
        r = self.da4.getVecArray(self.localR)
        p = self.da1.getVecArray(self.localP)
        
        cdef np.ndarray[np.float64_t, ndim=3] y = self.da4.getVecArray(Y)[...]

        cdef np.ndarray[np.float64_t, ndim=2] RBx = r[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] RBy = r[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] RVx = r[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] RVy = r[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] RP  = r[...][:,:,4]
        cdef np.ndarray[np.float64_t, ndim=2] dP  = p[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                # B_x
                y[iy, jy, 0] = - self.ht * RBx[ix, jx]
                
                # B_y
                y[iy, jy, 1] = - self.ht * RBy[ix, jx]
                
                # V_x
                y[iy, jy, 2] = - self.ht * (RVx[ix, jx] + self.derivatives.gradx(dP, ix, jx))
                
                # V_y
                y[iy, jy, 3] = - self.ht * (RVy[ix, jx] + self.derivatives.grady(dP, ix, jx))
                
                # P
                y[iy, jy, 4] = dP[ix, jx]
                
            

