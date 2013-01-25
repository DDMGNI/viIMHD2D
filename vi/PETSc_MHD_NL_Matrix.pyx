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



cdef class PETScMatrix(object):
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
        
        # factors of derivatives
        self.fac_dudt  = 1.   / 16. / self.ht
        self.fac_fdudx = 0.25 / 32. / self.hx
        self.fac_fdudy = 0.25 / 32. / self.hy
        
        # relaxation parameter
        self.omega = omega
        
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
    def formMat(self, Mat A, Mat P=None):
        cdef np.int64_t i, j, ia, ja, ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
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
        
        cdef np.ndarray[np.float64_t, ndim=2] A_arr


        A.zeroEntries()
        
        if P != None:
            P.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                
                # V_x
                row.index = (i,j)
                row.field = 0
                
                # dt(Vx)
                # + fdudx(Vxp, Vx)
                # + fdudx(Vxh, Vx)
                # + fdudy(Vyp, Vx)
                # + fdudy(Vyh, Vx)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.fdudx(A_arr, Vxp, ix, jx, +1)
                self.fdudx(A_arr, Vxh, ix, jx, +1)
                self.fdudy(A_arr, Vyp, ix, jx, +1)
                self.fdudy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # - fdudx(Bxp, Bx)
                # - fdudx(Bxh, Bx)
                # - fdudy(Byp, Bx)
                # - fdudy(Byh, Bx)
                
                A_arr = np.zeros((3,3))
                
                self.fdudx(A_arr, Bxp, ix, jx, -1)
                self.fdudx(A_arr, Bxh, ix, jx, -1)
                self.fdudy(A_arr, Byp, ix, jx, -1)
                self.fdudy(A_arr, Byh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # V_y
                row.index = (i,j)
                row.field = 1
                
                # dt(Vy)
                # + fdudx(Vxp, Vy)
                # + fdudx(Vxh, Vy)
                # + fdudy(Vyp, Vy)
                # + fdudy(Vyh, Vy)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.fdudx(A_arr, Vxp, ix, jx, +1)
                self.fdudx(A_arr, Vxh, ix, jx, +1)
                self.fdudy(A_arr, Vyp, ix, jx, +1)
                self.fdudy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # - fdudx(Bxp, By)
                # - fdudx(Bxh, By)
                # - fdudy(Byp, By)
                # - fdudy(Byh, By)
                
                A_arr = np.zeros((3,3))
                
                self.fdudx(A_arr, Bxp, ix, jx, -1)
                self.fdudx(A_arr, Bxh, ix, jx, -1)
                self.fdudy(A_arr, Byp, ix, jx, -1)
                self.fdudy(A_arr, Byh, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # B_x
                row.index = (i,j)
                row.field = 2
                
                # - fdudx(Bxp, Vx)
                # - fdudx(Bxh, Vx)
                # - fdudy(Byp, Vx)
                # - fdudy(Byh, Vx)
                
                A_arr = np.zeros((3,3))
                
                self.fdudx(A_arr, Bxp, ix, jx, -1)
                self.fdudx(A_arr, Bxh, ix, jx, -1)
                self.fdudy(A_arr, Byp, ix, jx, -1)
                self.fdudy(A_arr, Byh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])

                
                # dt(Bx)
                # + fdudx(Vxp, Bx)
                # + fdudx(Vxh, Bx)
                # + fdudy(Vyp, Bx)
                # + fdudy(Vyh, Bx)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.fdudx(A_arr, Vxp, ix, jx, +1)
                self.fdudx(A_arr, Vxh, ix, jx, +1)
                self.fdudy(A_arr, Vyp, ix, jx, +1)
                self.fdudy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # B_y
                row.index = (i,j)
                row.field = 3
                
                # - fdudx(Bxp, Vy)
                # - fdudx(Bxh, Vy)
                # - fdudy(Byp, Vy)
                # - fdudy(Byh, Vy)
                
                A_arr = np.zeros((3,3))
                
                self.fdudx(A_arr, Bxp, ix, jx, -1)
                self.fdudx(A_arr, Bxh, ix, jx, -1)
                self.fdudy(A_arr, Byp, ix, jx, -1)
                self.fdudy(A_arr, Byh, ix, jx, -1)
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(By)
                # + fdudx(Vxp, By)
                # + fdudx(Vxh, By)
                # + fdudy(Vyp, By)
                # + fdudy(Vyh, By)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.fdudx(A_arr, Vxp, ix, jx, +1)
                self.fdudx(A_arr, Vxh, ix, jx, +1)
                self.fdudy(A_arr, Vyp, ix, jx, +1)
                self.fdudy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 3
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])

                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
        A.assemble()
        
        if P != None:
            P.assemble()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Matrix")
        
                
    
    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xs, xe, ys, ye
        
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
        
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Vxp = xp[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vyp = xp[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bxp = xp[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Byp = xp[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Ph  = ph[...]
        cdef np.ndarray[np.float64_t, ndim=2] Pp  = pp[...]
        
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                # V_x
                b[iy, jy, 0] = self.derivatives.dt(Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxp, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxh, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyp, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyh, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxp, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxh, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byp, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byh, Bxh, ix, jx) \
                             - 0.5 * self.derivatives.gradx(Pp, ix, jx) \
                             - 0.5 * self.derivatives.gradx(Ph, ix, jx)

                # V_y
                b[iy, jy, 1] = self.derivatives.dt(Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxp, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyp, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxp, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxh, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byp, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byh, Byh, ix, jx) \
                             - 0.5 * self.derivatives.grady(Pp, ix, jx) \
                             - 0.5 * self.derivatives.grady(Ph, ix, jx)
                
                # B_x
                b[iy, jy, 2] = self.derivatives.dt(Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxp, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxh, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyp, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyh, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxp, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxh, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byp, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byh, Vxh, ix, jx)
                
                # B_y
                b[iy, jy, 3] = self.derivatives.dt(Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxp, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxh, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyp, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyh, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxp, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byp, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byh, Vyh, ix, jx)
                
    
    
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     RHS")
    
    
    @cython.boundscheck(False)
    def pressure(self, Vec X, Vec Y):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da4.globalToLocal(X, self.localX)
        
        x = self.da4.getVecArray(self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=2] P  = self.da1.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Pp = self.da1.getVecArray(self.Pp)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] By = x[...][:,:,3]
        
        
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

    
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Pressure")



    @cython.boundscheck(False)
    def timestep(self, np.ndarray[np.float64_t, ndim=3] x,
                       np.ndarray[np.float64_t, ndim=2] p,
                       np.ndarray[np.float64_t, ndim=3] y):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
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
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] A,
                               np.uint64_t i, np.uint64_t j):
        
        # (i-1, j-1)
        A[0,0] += 1. * self.fac_dudt
        
        # (i-1, j  )
        A[0,1] += 2. * self.fac_dudt
        
        # (i-1, j+1)
        A[0,2] += 1. * self.fac_dudt
        
        # (i,   j-1)
        A[1,0] += 2. * self.fac_dudt
        
        # (i,   j  )
        A[1,1] += 4. * self.fac_dudt
        
        # (i,   j+1)
        A[1,2] += 2. * self.fac_dudt
        
        # (i+1, j-1)
        A[2,0] += 1. * self.fac_dudt
        
        # (i+1, j  )
        A[2,1] += 2. * self.fac_dudt
        
        # (i+1, j+1)
        A[2,2] += 1. * self.fac_dudt
        
        

    @cython.boundscheck(False)
    cdef np.float64_t fdudx(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign):
        
        # (i-1, j-1)
        A[0,0] += - 1. * ( F[i-1, j  ] + F[i-1, j-1]) * self.fac_fdudx * sign \
                  - 1. * ( F[i,   j  ] + F[i,   j-1]) * self.fac_fdudx * sign
        
        # (i-1, j  )
        A[0,1] += - 1. * ( F[i-1, j-1] + F[i,   j-1]) * self.fac_fdudx * sign \
                  - 2. * ( F[i-1, j  ] + F[i,   j  ]) * self.fac_fdudx * sign \
                  - 1. * ( F[i-1, j+1] + F[i,   j+1]) * self.fac_fdudx * sign
        
        # (i-1, j+1)
        A[0,2] += - 1. * ( F[i-1, j  ] + F[i-1, j+1]) * self.fac_fdudx * sign \
                  - 1. * ( F[i,   j  ] + F[i,   j+1]) * self.fac_fdudx * sign
        
        # (i,   j-1)
        A[1,0] += - 1. * ( F[i+1, j-1] - F[i-1, j-1]) * self.fac_fdudx * sign \
                  - 1. * ( F[i+1, j  ] - F[i-1, j  ]) * self.fac_fdudx * sign
        
        # (i,   j  )
        A[1,1] += - 1. * ( F[i+1, j-1] - F[i-1, j-1]) * self.fac_fdudx * sign \
                  - 2. * ( F[i+1, j  ] - F[i-1, j  ]) * self.fac_fdudx * sign \
                  - 1. * ( F[i+1, j+1] - F[i-1, j+1]) * self.fac_fdudx * sign
        
        # (i,   j+1)
        A[1,2] += - 1. * ( F[i+1, j  ] - F[i-1, j  ]) * self.fac_fdudx * sign \
                  - 1. * ( F[i+1, j+1] - F[i-1, j+1]) * self.fac_fdudx * sign
        
        # (i+1, j-1)
        A[2,0] += + 1. * ( F[i,   j-1] + F[i,   j  ]) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j-1] + F[i+1, j  ]) * self.fac_fdudx * sign
        
        # (i+1, j  )
        A[2,1] += + 1. * ( F[i,   j-1] + F[i+1, j-1]) * self.fac_fdudx * sign \
                  + 2. * ( F[i,   j  ] + F[i+1, j  ]) * self.fac_fdudx * sign \
                  + 1. * ( F[i,   j+1] + F[i+1, j+1]) * self.fac_fdudx * sign
        
        # (i+1, j+1)
        A[2,2] += + 1. * ( F[i,   j  ] + F[i,   j+1]) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j  ] + F[i+1, j+1]) * self.fac_fdudx * sign
        
        

    @cython.boundscheck(False)
    cdef np.float64_t fdudy(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign):
        
        # (i-1, j-1)
        A[0,0] += - 1. * ( F[i-1, j-1] + F[i-1, j  ]) * self.fac_fdudy * sign \
                  - 1. * ( F[i,   j-1] + F[i,   j  ]) * self.fac_fdudy * sign
        
        # (i-1, j  )
        A[0,1] += - 1. * ( F[i-1, j+1] - F[i-1, j-1]) * self.fac_fdudy * sign \
                  - 1. * ( F[i,   j+1] - F[i,   j-1]) * self.fac_fdudy * sign
        
        # (i-1, j+1)
        A[0,2] += + 1. * ( F[i-1, j  ] + F[i-1, j+1]) * self.fac_fdudy * sign \
                  + 1. * ( F[i,   j  ] + F[i,   j+1]) * self.fac_fdudy * sign
        
        # (i,   j-1)
        A[1,0] += - 1. * ( F[i-1, j-1] + F[i-1, j  ]) * self.fac_fdudy * sign \
                  - 2. * ( F[i,   j-1] + F[i,   j  ]) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j-1] + F[i+1, j  ]) * self.fac_fdudy * sign
        
        # (i,   j  )
        A[1,1] += - 1. * ( F[i-1, j+1] - F[i-1, j-1]) * self.fac_fdudy * sign \
                  - 2. * ( F[i,   j+1] - F[i,   j-1]) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j+1] - F[i+1, j-1]) * self.fac_fdudy * sign
        
        # (i,   j+1)
        A[1,2] += + 1. * ( F[i-1, j  ] + F[i-1, j+1]) * self.fac_fdudy * sign \
                  + 2. * ( F[i,   j  ] + F[i,   j+1]) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j  ] + F[i+1, j+1]) * self.fac_fdudy * sign
        
        # (i+1, j-1)
        A[2,0] += - 1. * ( F[i,   j-1] + F[i,   j  ]) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j-1] + F[i+1, j  ]) * self.fac_fdudy * sign
        
        # (i+1, j  )
        A[2,1] += - 1. * ( F[i,   j+1] - F[i,   j-1]) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j+1] - F[i+1, j-1]) * self.fac_fdudy * sign
        
        # (i+1, j+1)
        A[2,2] += + 1. * ( F[i,   j  ] + F[i,   j+1]) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j  ] + F[i+1, j+1]) * self.fac_fdudy * sign
        
        
