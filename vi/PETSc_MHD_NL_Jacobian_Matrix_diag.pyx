'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport DMDA, Mat, Vec

from PETSc_MHD_Derivatives import  PETSc_MHD_Derivatives
from PETSc_MHD_Derivatives cimport PETSc_MHD_Derivatives



cdef class PETScJacobian(object):
    '''
    
    '''
    
    def __init__(self, DMDA da1, DMDA da4,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
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
        
        # factors of derivatives
        self.fac_dx    = 0.5 / 4. / 2. / self.hx
        self.fac_dy    = 0.5 / 4. / 2. / self.hy
        self.fac_dudt  = 1.   / 16. / self.ht
        self.fac_fdudx = 0.25 / 32. / self.hx
        self.fac_fdudy = 0.25 / 32. / self.hy
        
        # create history vectors
        self.Xh = self.da4.createGlobalVec()
        self.Xp = self.da4.createGlobalVec()
        self.Ph = self.da1.createGlobalVec()
        self.Pp = self.da1.createGlobalVec()
        
        # create local vectors
        self.localX  = da4.createLocalVec()
        self.localXh = da4.createLocalVec()
        self.localXp = da4.createLocalVec()
        self.localPh = da1.createLocalVec()
        self.localPp = da1.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X, Vec P):
        X.copy(self.Xh)
        P.copy(self.Ph)
        
    
    def update_previous(self, Vec X, Vec P):
        X.copy(self.Xp)
        P.copy(self.Pp)
        
    
    @cython.boundscheck(False)
    def formMatA(self, Mat A, Mat Ainv=None):
        cdef np.int64_t i, j, ix, jx, index
        cdef np.int64_t xe, xs, ye, ys
        
        cdef np.float64_t value
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da4.globalToLocal(self.Xh, self.localXh)
        self.da4.globalToLocal(self.Xp, self.localXp)
        
        xh = self.da4.getVecArray(self.localXh)
        xp = self.da4.getVecArray(self.localXp)
        
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Vxp = xp[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vyp = xp[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bxp = xp[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Byp = xp[...][:,:,3]
        
        
        A.zeroEntries()
        
        if Ainv != None:
            Ainv.zeroEntries()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                
                # V_x

                # dt(dVx)
                # + fdudx(Vxp, dVx)
                # + fdudx(Vxh, dVx)
                # + fdudy(Vyp, dVx)
                # + fdudy(Vyh, dVx)
                # + udfdx(dVx, Vxp)
                # + udfdx(dVx, Vxh)
                
                index = 0 * self.nx * self.ny \
                      + i + j * self.nx
                
                value = 4. * self.fac_dudt \
                      + self.fdudx(Vxp, ix, jx) \
                      + self.fdudx(Vxh, ix, jx) \
                      + self.fdudy(Vyp, ix, jx) \
                      + self.fdudy(Vyh, ix, jx) \
                      + self.udfdx(Vxp, ix, jx) \
                      + self.udfdx(Vxh, ix, jx)
                
                A.setValue(index, index, value)
                
                if Ainv != None:
                    Ainv.setValue(index, index, 1. / value)
                
                
                # V_y
                
                # dt(Vy)
                # + fdudx(Vxp, Vy)
                # + fdudx(Vxh, Vy)
                # + fdudy(Vyp, Vy)
                # + fdudy(Vyh, Vy)
                # + udfdy(dVy, Vyp)
                # + udfdy(dVy, Vyh)
                
                index = 1 * self.nx * self.ny \
                      + i + j * self.nx
                
                value = 4. * self.fac_dudt \
                      + self.fdudx(Vxp, ix, jx) \
                      + self.fdudx(Vxh, ix, jx) \
                      + self.fdudy(Vyp, ix, jx) \
                      + self.fdudy(Vyh, ix, jx) \
                      + self.udfdy(Vyp, ix, jx) \
                      + self.udfdy(Vyh, ix, jx)
                
                A.setValue(index, index, value)
                
                if Ainv != None:
                    Ainv.setValue(index, index, 1. / value)
                
                
                # B_x
                
                # dt(dBx)
                # + fdudx(Vxp, dBx)
                # + fdudx(Vxh, dBx)
                # + fdudy(Vyp, dBx)
                # + fdudy(Vyh, dBx)
                # - udfdx(dBx, Vxp)
                # - udfdx(dBx, Vxh)
                
                index = 2 * self.nx * self.ny \
                      + i + j * self.nx
                
                value = 4. * self.fac_dudt \
                      + self.fdudx(Vxp, ix, jx) \
                      + self.fdudx(Vxh, ix, jx) \
                      + self.fdudy(Vyp, ix, jx) \
                      + self.fdudy(Vyh, ix, jx) \
                      - self.udfdx(Vxp, ix, jx) \
                      - self.udfdx(Vxh, ix, jx)
                
                A.setValue(index, index, value)
                
                if Ainv != None:
                    Ainv.setValue(index, index, 1. / value)
                
                
                # B_y
                
                # dt(dBy)
                # + fdudx(Vxp, dBy)
                # + fdudx(Vxh, dBy)
                # + fdudy(Vyp, dBy)
                # + fdudy(Vyh, dBy)
                # - udfdy(dBy, Vyp)
                # - udfdy(dBy, Vyh)

                index = 3 * self.nx * self.ny \
                      + i + j * self.nx
                
                value = 4. * self.fac_dudt \
                      + self.fdudx(Vxp, ix, jx) \
                      + self.fdudx(Vxh, ix, jx) \
                      + self.fdudy(Vyp, ix, jx) \
                      + self.fdudy(Vyh, ix, jx) \
                      - self.udfdy(Vyp, ix, jx) \
                      - self.udfdy(Vyh, ix, jx)
                
                A.setValue(index, index, value)
                
                if Ainv != None:
                    Ainv.setValue(index, index, 1. / value)
                
        
        A.assemble()
        
        if Ainv != None:
            Ainv.assemble()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Matrix")
        
                
    
    @cython.boundscheck(False)
    def formMatB(self, Mat B not None):
        cdef np.int64_t i, j, ti, tj
        cdef np.int64_t row, col, tcol
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        B.zeroEntries()
        
        
        for i in np.arange(xs, xe):
            for j in np.arange(ys, ye):
                col = i + j * self.nx
                
                # dx
                row = col 
                
                for ti, tj, value in [
                        (-1, -1, - 1. * self.fac_dx),
                        (+1, -1, + 1. * self.fac_dx),
                        (-1,  0, - 2. * self.fac_dx),
                        (+1,  0, + 2. * self.fac_dx),
                        (-1, +1, - 1. * self.fac_dx),
                        (+1, +1, + 1. * self.fac_dx),
                    ]:
                    
                    tcol = (col + ti + tj * self.nx + self.nx * self.ny) % (self.nx * self.ny)
                    B.setValue(row, tcol, value)
                
                
                # dy
                row = col + self.nx * self.ny
                
                for ti, tj, value in [
                        (-1, -1, - 1. * self.fac_dy),
                        (-1, +1, + 1. * self.fac_dy),
                        ( 0, -1, - 2. * self.fac_dy),
                        ( 0, +1, + 2. * self.fac_dy),
                        (+1, -1, - 1. * self.fac_dy),
                        (+1, +1, + 1. * self.fac_dy),
                    ]:
        
                    tcol = (col + ti + tj * self.nx + self.nx * self.ny) % (self.nx * self.ny)
                    B.setValue(row, tcol, value)
        
        
        B.assemble()
        
        

    @cython.boundscheck(False)
    def timestep(self, np.ndarray[np.float64_t, ndim=3] x,
                       np.ndarray[np.float64_t, ndim=2] p,
                       np.ndarray[np.float64_t, ndim=3] y):
        
        cdef np.int64_t ix, iy, i, j
        cdef np.int64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] By = x[:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] P  = p[:,:]
        
        
        for j in np.arange(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
                # V_x
                y[iy, jy, 0] = \
                             - self.derivatives.fdudx(Vx, Vx, ix, jx) \
                             - self.derivatives.fdudy(Vy, Vx, ix, jx) \
                             + self.derivatives.fdudx(Bx, Bx, ix, jx) \
                             + self.derivatives.fdudy(By, Bx, ix, jx) \
                             - self.derivatives.gradx(P, ix, jx)
                              
                # V_y
                y[iy, jy, 1] = \
                             - self.derivatives.fdudx(Vx, Vy, ix, jx) \
                             - self.derivatives.fdudy(Vy, Vy, ix, jx) \
                             + self.derivatives.fdudx(Bx, By, ix, jx) \
                             + self.derivatives.fdudy(By, By, ix, jx) \
                             - self.derivatives.grady(P, ix, jx)
          
                # B_x
                y[iy, jy, 2] = \
                             - self.derivatives.fdudx(Vx, Bx, ix, jx) \
                             - self.derivatives.fdudy(Vy, Bx, ix, jx) \
                             + self.derivatives.fdudx(Bx, Vx, ix, jx) \
                             + self.derivatives.fdudy(By, Vx, ix, jx)
                    
                # B_y
                y[iy, jy, 3] = \
                             - self.derivatives.fdudx(Vx, By, ix, jx) \
                             - self.derivatives.fdudy(Vy, By, ix, jx) \
                             + self.derivatives.fdudx(Bx, Vy, ix, jx) \
                             + self.derivatives.fdudy(By, Vy, ix, jx)
                                


    @cython.boundscheck(False)
    cdef np.float64_t fdudx(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t value
        
        value = - 1. * ( F[i+1, j-1] - F[i-1, j-1] ) \
                - 2. * ( F[i+1, j  ] - F[i-1, j  ] ) \
                - 1. * ( F[i+1, j+1] - F[i-1, j+1] )
        
        return value * self.fac_fdudx
        
        

    @cython.boundscheck(False)
    cdef np.float64_t fdudy(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t value
        
        value = - 1. * ( F[i-1, j+1] - F[i-1, j-1] ) \
                - 2. * ( F[i,   j+1] - F[i,   j-1] ) \
                - 1. * ( F[i+1, j+1] - F[i+1, j-1] )
        
        return value * self.fac_fdudy
        
        
    @cython.boundscheck(False)
    cdef np.float64_t udfdx(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t value
        
        value = + 1. * ( F[i+1, j-1] - F[i-1, j-1] ) \
                + 2. * ( F[i+1, j  ] - F[i-1, j  ] ) \
                + 1. * ( F[i+1, j+1] - F[i-1, j+1] )
        
        return value * self.fac_fdudx
        
        
    
    @cython.boundscheck(False)
    cdef np.float64_t udfdy(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t value
        
        value = + 1. * ( F[i-1, j+1] - F[i-1, j-1] ) \
                + 2. * ( F[i,   j+1] - F[i,   j-1] ) \
                + 1. * ( F[i+1, j+1] - F[i+1, j-1] )
        
        return value * self.fac_fdudy
