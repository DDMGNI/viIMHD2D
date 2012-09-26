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

from PETSc_MHD_Derivatives import  PETSc_MHD_Derivatives
from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScSolver(object):
    '''
    
    '''
    
    def __init__(self, DA da1, DA da4,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy,
                 np.float64_t omega):
        '''
        Constructor
        '''
        
        assert da1.getDim() == 2
        assert da4.getDim() == 2
        
        # distributed array
        self.da1 = da1
        self.da4 = da4
        
        # grid size
        self.nx = nx
        self.ny = ny
        
        # step size
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        # relaxation parameter
        self.omega = omega
        
        # create history vectors
        self.Xh = self.da4.createGlobalVec()
        self.Xp = self.da4.createGlobalVec()
        self.Ph = self.da1.createGlobalVec()
        self.Pp = self.da1.createGlobalVec()
        
        # create local vectors
        self.localB  = da4.createLocalVec()
        self.localX  = da4.createLocalVec()
        self.localXh = da4.createLocalVec()
        self.localXp = da4.createLocalVec()
        self.localPh = da1.createLocalVec()
        self.localPp = da1.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X, Vec P):
        x  = self.da4.getVecArray(X)
        p  = self.da1.getVecArray(P)
        xh = self.da4.getVecArray(self.Xh)
        ph = self.da1.getVecArray(self.Ph)
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        xh[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        ph[xs:xe, ys:ye]    = p[xs:xe, ys:ye]
        
    
    def update_previous(self, Vec X, Vec P):
        x  = self.da4.getVecArray(X)
        p  = self.da1.getVecArray(P)
        xp = self.da4.getVecArray(self.Xp)
        pp = self.da1.getVecArray(self.Pp)
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        xp[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        pp[xs:xe, ys:ye]    = p[xs:xe, ys:ye]
        
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        cdef np.float64_t meanDivV
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(X,       self.localX)
        
        x = self.da4.getVecArray(self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=3] y = self.da4.getVecArray(Y)[...]

        cdef np.ndarray[np.float64_t, ndim=2] Bx  = x[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By  = x[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx  = x[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy  = x[...][:,:,3]
        

                
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                # B_x
                y[iy, jy, 0] = self.derivatives.dt(Bx, ix, jx)
                    
                # B_y
                y[iy, jy, 1] = self.derivatives.dt(By, ix, jx)
                
                # V_x
                y[iy, jy, 2] = self.derivatives.dt(Vx, ix, jx)
                              
                # V_y
                y[iy, jy, 3] = self.derivatives.dt(Vy, ix, jx)
        
        
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(self.Xh, self.localXh)
        self.da4.globalToLocal(self.Xp, self.localXp)
        self.da1.globalToLocal(self.Ph, self.localPh)
        self.da1.globalToLocal(self.Pp, self.localPp)
        
        xh = self.da4.getVecArray(self.localXh)
        xp = self.da4.getVecArray(self.localXp)
        ph = self.da1.getVecArray(self.localPh)
        pp = self.da1.getVecArray(self.localPp)
        
        cdef np.ndarray[np.float64_t, ndim=3] b = self.da4.getVecArray(B)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxp = xp[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byp = xp[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxp = xp[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyp = xp[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Ph  = ph[...]
        cdef np.ndarray[np.float64_t, ndim=2] Pp  = pp[...]
        
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                b[iy, jy, 0] = self.derivatives.dt(Bxh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Bxp, Vyp, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Bxp, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Bxh, Vyp, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Bxh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Byp, Vxp, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Byp, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Byh, Vxp, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Byh, Vxh, ix, jx) \
                
                b[iy, jy, 1] = self.derivatives.dt(Byh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Byp, Vxp, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Byp, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Byh, Vxp, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Byh, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxp, Vyp, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxp, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxh, Vyp, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxh, Vyh, ix, jx)
                
                b[iy, jy, 2] = self.derivatives.dt(Vxh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Vxp, Vxp, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Vxp, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Vxh, Vxp, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Vxh, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Vxp, Vyp, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Vxp, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Vxh, Vyp, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Vxh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxp, Bxp, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxp, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxh, Bxp, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxh, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Bxp, Byp, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Bxp, Byh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Bxh, Byp, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Bxh, Byh, ix, jx) \
                             - 0.5  * self.derivatives.gradx(Pp, ix, jx) \
                             - 0.5  * self.derivatives.gradx(Ph, ix, jx)

                b[iy, jy, 3] = self.derivatives.dt(Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Vxp, Vyp, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Vxp, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Vxh, Vyp, ix, jx) \
                             - 0.25 * self.derivatives.dx1(Vxh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Vyp, Vyp, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Vyp, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Vyh, Vyp, ix, jx) \
                             - 0.25 * self.derivatives.dy1(Vyh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxp, Byp, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxp, Byh, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxh, Byp, ix, jx) \
                             + 0.25 * self.derivatives.dx1(Bxh, Byh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Byp, Byp, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Byp, Byh, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Byh, Byp, ix, jx) \
                             + 0.25 * self.derivatives.dy1(Byh, Byh, ix, jx) \
                             - 0.5  * self.derivatives.grady(Pp, ix, jx) \
                             - 0.5  * self.derivatives.grady(Ph, ix, jx)
                
    
    
#    @cython.boundscheck(False)
    def pressure(self, Vec X, Vec Y):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da4.globalToLocal(X, self.localX)
        
        x = self.da4.getVecArray(self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=2] P  = self.da1.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Pp = self.da1.getVecArray(self.Pp)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By = x[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[...][:,:,3]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                P[iy, jy] = Pp[iy, jy] \
                          - self.omega * ( \
                             + self.derivatives.gradx(Vx, ix, jx) \
                             + self.derivatives.grady(Vy, ix, jx) \
                          )


##    @cython.boundscheck(False)
#    def timestep(self, np.ndarray[np.float64_t, ndim=3] x,
#                       np.ndarray[np.float64_t, ndim=3] y):
#        
#        cdef np.uint64_t ix, iy, i, j
#        cdef np.uint64_t xs, xe, ys, ye
#        
#        (xs, xe), (ys, ye) = self.da4.getRanges()
#        
#        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[:,:,0]
#        cdef np.ndarray[np.float64_t, ndim=2] By = x[:,:,1]
#        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[:,:,2]
#        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[:,:,3]
#        cdef np.ndarray[np.float64_t, ndim=2] P  = x[:,:,4]
#        
#        
#        for j in np.arange(ys, ye):
#            jx = j-ys+1
#            jy = j-ys
#            
#            for i in np.arange(xs, xe):
#                ix = i-xs+1
#                iy = i-xs
#                
#                # B_x
#                y[iy, jy, 0] = \
#                             - self.derivatives.dy1(Bx, Vy, ix, jx) \
#                             + self.derivatives.dy1(By, Vx, ix, jx)
#                    
#                # B_y
#                y[iy, jy, 1] = \
#                             - self.derivatives.dx1(By, Vx, ix, jx) \
#                             + self.derivatives.dx1(Bx, Vy, ix, jx)
#                                
#                # V_x
#                y[iy, jy, 2] = \
#                             - self.derivatives.dx1(Vx, Vx, ix, jx) \
#                             - self.derivatives.dy1(Vx, Vy, ix, jx) \
#                             + self.derivatives.dx1(Bx, Bx, ix, jx) \
#                             + self.derivatives.dy1(Bx, By, ix, jx) \
#                             - self.derivatives.gradx(P, ix, jx)
#                              
#                # V_y
#                y[iy, jy, 3] = \
#                             - self.derivatives.dx1(Vx, Vy, ix, jx) \
#                             - self.derivatives.dy1(Vy, Vy, ix, jx) \
#                             + self.derivatives.dx1(Bx, By, ix, jx) \
#                             + self.derivatives.dy1(By, By, ix, jx) \
#                             - self.derivatives.grady(P, ix, jx)
#                              
#                # P
#                y[iy, jy, 4] = 0.0
#          
