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
        
        # create history vector
        self.Xh = self.da4.createGlobalVec()
        self.Xp = self.da4.createGlobalVec()
        
        # create local vectors
        self.localX  = da4.createLocalVec()
        self.localXh = da4.createLocalVec()
        self.localXp = da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        x  = self.da4.getVecArray(X)
        xh = self.da4.getVecArray(self.Xh)
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        xh[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        
    
    def update_previous(self, Vec X):
        x  = self.da4.getVecArray(X)
        xp = self.da4.getVecArray(self.Xp)
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        xp[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(X,       self.localX)
        self.da4.globalToLocal(self.Xh, self.localXh)
        self.da4.globalToLocal(self.Xp, self.localXp)
        
        dx = self.da4.getVecArray(self.localX)
        xh = self.da4.getVecArray(self.localXh)
        xp = self.da4.getVecArray(self.localXp)
        
        cdef np.ndarray[np.float64_t, ndim=3] y = self.da4.getVecArray(Y)[...]

        cdef np.ndarray[np.float64_t, ndim=2] dBx = dx[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] dBy = dx[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] dVx = dx[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] dVy = dx[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] dP  = dx[...][:,:,4]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxp = xp[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byp = xp[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxp = xp[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyp = xp[...][:,:,3]

                
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                # B_x
                y[iy, jy, 0] = self.derivatives.dt_diag(dBx, ix, jx)
#                y[iy, jy, 0] = self.derivatives.dt(dBx, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(dBx, Vyp, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(Bxp, dVy, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(dBx, Vyh, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(Bxh, dVy, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(dBy, Vxp, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(Byp, dVx, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(dBy, Vxh, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(Byh, dVx, ix, jx)
                    
                # B_y
                y[iy, jy, 1] = self.derivatives.dt_diag(dBy, ix, jx)
#                y[iy, jy, 1] = self.derivatives.dt(dBy, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(dBy, Vxp, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(Byp, dVx, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(dBy, Vxh, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(Byh, dVx, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(dBx, Vyp, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(Bxp, dVy, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(dBx, Vyh, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(Bxh, dVy, ix, jx)
                
                # V_x
                y[iy, jy, 2] = self.derivatives.dt_diag(dVx, ix, jx) \
                             + 0.5 * self.derivatives.gradx(dP, ix, jx)
#                y[iy, jy, 2] = self.derivatives.dt(dVx, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(dVx, Vxp, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(Vxp, dVx, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(dVx, Vxh, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(Vxh, dVx, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(dVx, Vyp, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(Vxp, dVy, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(dVx, Vyh, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(Vxh, dVy, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(dBx, Bxp, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(Bxp, dBx, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(dBx, Bxh, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(Bxh, dBx, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(dBx, Byp, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(Bxp, dBy, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(dBx, Byh, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(Bxh, dBy, ix, jx) \
#                             + 0.5 * self.derivatives.gradx(dP, ix, jx)
                              
                # V_y
                y[iy, jy, 3] = self.derivatives.dt_diag(dVy, ix, jx) \
                             + 0.5 * self.derivatives.grady(dP, ix, jx)
#                y[iy, jy, 3] = self.derivatives.dt(dVy, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(dVx, Vyp, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(Vxp, dVy, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(dVx, Vyh, ix, jx) \
#                             + 0.25 * self.derivatives.dx1(Vxh, dVy, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(dVy, Vyp, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(Vyp, dVy, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(dVy, Vyh, ix, jx) \
#                             + 0.25 * self.derivatives.dy1(Vyh, dVy, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(dBx, Byp, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(Bxp, dBy, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(dBx, Byh, ix, jx) \
#                             - 0.25 * self.derivatives.dx1(Bxh, dBy, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(dBy, Byp, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(Byp, dBy, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(dBy, Byh, ix, jx) \
#                             - 0.25 * self.derivatives.dy1(Byh, dBy, ix, jx) \
#                             + 0.5 * self.derivatives.grady(dP, ix, jx)
                              
                # P
                y[iy, jy, 4] = 0.5 * self.derivatives.gradx(dVx, ix, jx) \
                             + 0.5 * self.derivatives.grady(dVy, ix, jx)
        
