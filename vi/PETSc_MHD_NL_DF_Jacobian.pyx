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



cdef class PETScJacobian(object):
    '''
    
    '''
    
    def __init__(self, DA da1, DA da5,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        assert da1.getDim() == 2
        assert da5.getDim() == 2
        
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
        self.Xp = self.da5.createGlobalVec()
        
        # create local vectors
        self.localX  = da5.createLocalVec()
        self.localXh = da5.createLocalVec()
        self.localXp = da5.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da5.globalToLocal(X,       self.localX)
        self.da5.globalToLocal(self.Xh, self.localXh)
        self.da5.globalToLocal(self.Xp, self.localXp)
        
        dx = self.da5.getVecArray(self.localX)
        xh = self.da5.getVecArray(self.localXh)
        xp = self.da5.getVecArray(self.localXp)
        
        cdef np.ndarray[np.float64_t, ndim=3] y = self.da5.getVecArray(Y)[...]

        cdef np.ndarray[np.float64_t, ndim=2] dVx = dx[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] dVy = dx[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] dBx = dx[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] dBy = dx[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] dP  = dx[...][:,:,4]
        
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Vxp = xp[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vyp = xp[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bxp = xp[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Byp = xp[...][:,:,3]

                
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = self.derivatives.dt(dVx, ix, jx) \
                             + 0.25 * self.derivatives.dx(dVx, Vxp, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxp, dVx, ix, jx) \
                             + 0.25 * self.derivatives.dx(dVx, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxh, dVx, ix, jx) \
                             + 0.25 * self.derivatives.dy(dVx, Vyp, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vxp, dVy, ix, jx) \
                             + 0.25 * self.derivatives.dy(dVx, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vxh, dVy, ix, jx) \
                             - 0.25 * self.derivatives.dx(dBx, Bxp, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxp, dBx, ix, jx) \
                             - 0.25 * self.derivatives.dx(dBx, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxh, dBx, ix, jx) \
                             - 0.25 * self.derivatives.dy(dBx, Byp, ix, jx) \
                             - 0.25 * self.derivatives.dy(Bxp, dBy, ix, jx) \
                             - 0.25 * self.derivatives.dy(dBx, Byh, ix, jx) \
                             - 0.25 * self.derivatives.dy(Bxh, dBy, ix, jx) \
                             + 0.5 * self.derivatives.gradx(dP, ix, jx)
                              
                # V_y
                y[iy, jy, 1] = self.derivatives.dt(dVy, ix, jx) \
                             + 0.25 * self.derivatives.dx(dVx, Vyp, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxp, dVy, ix, jx) \
                             + 0.25 * self.derivatives.dx(dVx, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dx(Vxh, dVy, ix, jx) \
                             + 0.25 * self.derivatives.dy(dVy, Vyp, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vyp, dVy, ix, jx) \
                             + 0.25 * self.derivatives.dy(dVy, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy(Vyh, dVy, ix, jx) \
                             - 0.25 * self.derivatives.dx(dBx, Byp, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxp, dBy, ix, jx) \
                             - 0.25 * self.derivatives.dx(dBx, Byh, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxh, dBy, ix, jx) \
                             - 0.25 * self.derivatives.dy(dBy, Byp, ix, jx) \
                             - 0.25 * self.derivatives.dy(Byp, dBy, ix, jx) \
                             - 0.25 * self.derivatives.dy(dBy, Byh, ix, jx) \
                             - 0.25 * self.derivatives.dy(Byh, dBy, ix, jx) \
                             + 0.5 * self.derivatives.grady(dP, ix, jx)
                              
                # B_x
                y[iy, jy, 2] = self.derivatives.dt(dBx, ix, jx) \
                             + 0.25 * self.derivatives.dy(dBx, Vyp, ix, jx) \
                             + 0.25 * self.derivatives.dy(Bxp, dVy, ix, jx) \
                             + 0.25 * self.derivatives.dy(dBx, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy(Bxh, dVy, ix, jx) \
                             - 0.25 * self.derivatives.dy(dBy, Vxp, ix, jx) \
                             - 0.25 * self.derivatives.dy(Byp, dVx, ix, jx) \
                             - 0.25 * self.derivatives.dy(dBy, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.dy(Byh, dVx, ix, jx)
                    
                # B_y
                y[iy, jy, 3] = self.derivatives.dt(dBy, ix, jx) \
                             + 0.25 * self.derivatives.dx(dBy, Vxp, ix, jx) \
                             + 0.25 * self.derivatives.dx(Byp, dVx, ix, jx) \
                             + 0.25 * self.derivatives.dx(dBy, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dx(Byh, dVx, ix, jx) \
                             - 0.25 * self.derivatives.dx(dBx, Vyp, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxp, dVy, ix, jx) \
                             - 0.25 * self.derivatives.dx(dBx, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dx(Bxh, dVy, ix, jx)
                
                # P
                y[iy, jy, 4] = 0.5 * self.derivatives.gradx(dVx, ix, jx) \
                             + 0.5 * self.derivatives.grady(dVy, ix, jx)
        
