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
        
#        assert da1.getDim() == 2
#        assert da5.getDim() == 2
        
        # distributed array
        self.da1 = da1
        self.da5 = da5
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
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
        
        
#    @cython.boundscheck(False)
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
                y[iy, jy, 0] = self.derivatives.dt(Vx,  ix, jx) \
                             - self.derivatives.dt(Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vx,  Vx,  ix, jx) \
                             + 0.25 * self.derivatives.dx(Vx,  Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxh, Vx,  ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxh, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vy,  Vx,  ix, jx) \
                             + 0.25 * self.derivatives.dy(Vyh, Vx,  ix, jx) \
                             + 0.25 * self.derivatives.dy(Vy,  Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vyh, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bx,  Bx,  ix, jx) \
                             - 0.25 * self.derivatives.dx(Bx,  Bxh, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxh, Bx,  ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxh, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.dy(By,  Bx,  ix, jx) \
                             - 0.25 * self.derivatives.dy(Byh, Bx,  ix, jx) \
                             - 0.25 * self.derivatives.dy(By,  Bxh, ix, jx) \
                             - 0.25 * self.derivatives.dy(Byh, Bxh, ix, jx) \
                             + 0.5  * self.derivatives.gradx(P,  ix, jx) \
                             + 0.5  * self.derivatives.gradx(Ph, ix, jx)
                              
                # V_y
                y[iy, jy, 1] = self.derivatives.dt(Vy,  ix, jx) \
                             - self.derivatives.dt(Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vx,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dx(Vx,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vy,  Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dy(Vy,  Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vyh, Vy,  ix, jx) \
                             + 0.25 * self.derivatives.dy(Vyh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bx,  By,  ix, jx) \
                             - 0.25 * self.derivatives.dx(Bx,  Byh, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxh, By,  ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxh, Byh, ix, jx) \
                             - 0.25 * self.derivatives.dy(By,  By,  ix, jx) \
                             - 0.25 * self.derivatives.dy(By,  Byh, ix, jx) \
                             - 0.25 * self.derivatives.dy(Byh, By,  ix, jx) \
                             - 0.25 * self.derivatives.dy(Byh, Byh, ix, jx) \
                             + 0.5  * self.derivatives.grady(P,  ix, jx) \
                             + 0.5  * self.derivatives.grady(Ph, ix, jx)
                              
                # B_x
                y[iy, jy, 2] = self.derivatives.dt(Bx,  ix, jx) \
                             - self.derivatives.dt(Bxh, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vy,  Bx,  ix, jx) \
                             + 0.25 * self.derivatives.dy(Vyh, Bx,  ix, jx) \
                             + 0.25 * self.derivatives.dy(Vy,  Bxh, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vyh, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.dy(By,  Vx,  ix, jx) \
                             - 0.25 * self.derivatives.dy(By,  Vxh, ix, jx) \
                             - 0.25 * self.derivatives.dy(Byh, Vx,  ix, jx) \
                             - 0.25 * self.derivatives.dy(Byh, Vxh, ix, jx)
                    
                # B_y
                y[iy, jy, 3] = self.derivatives.dt(By,  ix, jx) \
                             - self.derivatives.dt(Byh, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vx,  By,  ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxh, By,  ix, jx) \
                             + 0.25 * self.derivatives.dx(Vx,  Byh, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxh, Byh, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bx,  Vy,  ix, jx) \
                             - 0.25 * self.derivatives.dx(Bx,  Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxh, Vy,  ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxh, Vyh, ix, jx)
                
                # P
                y[iy, jy, 4] = 0.5 * self.derivatives.gradx(Vx,  ix, jx) \
                             + 0.5 * self.derivatives.gradx(Vxh, ix, jx) \
                             + 0.5 * self.derivatives.grady(Vy,  ix, jx) \
                             + 0.5 * self.derivatives.grady(Vyh, ix, jx)
        
