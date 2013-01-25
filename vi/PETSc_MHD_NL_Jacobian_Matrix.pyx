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



cdef class PETScJacobian(object):
    '''
    
    '''
    
    def __init__(self, DA da1, DA da4,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy,
                 np.float64_t omega=0.0):
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
#        self.fac_dx    = 0.5 / 4. / 2. / self.hx
#        self.fac_dy    = 0.5 / 4. / 2. / self.hy
        self.fac_gradx = 0.5 / 4. / 2. / self.hx
        self.fac_grady = 0.5 / 4. / 2. / self.hy
#        self.fac_divx  = 1.0 / 4. / 2. / self.hx
#        self.fac_divy  = 1.0 / 4. / 2. / self.hy
        self.fac_divx  = 1.0 / self.hx
        self.fac_divy  = 1.0 / self.hy
        self.fac_dt  = 1.   / 16. / self.ht
        self.fac_fdudx = 0.25 / 32. / self.hx
        self.fac_fdudy = 0.25 / 32. / self.hy
        
        # relaxation parameter
        self.omega = omega
        
        # create history vectors
        self.Xh = self.da4.createGlobalVec()
#        self.Xp = self.da4.createGlobalVec()
#        self.Ph = self.da1.createGlobalVec()
#        self.Pp = self.da1.createGlobalVec()
        
        # create local vectors
        self.localX  = da4.createLocalVec()
#        self.localP  = da1.createLocalVec()
        self.localXh = da4.createLocalVec()
#        self.localPh = da1.createLocalVec()
#        self.localXp = da4.createLocalVec()
#        self.localPp = da1.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
#        P.copy(self.Ph)
        
    
#    def update_previous(self, Vec X, Vec P):
#        X.copy(self.Xp)
#        P.copy(self.Pp)
        
    
    @cython.boundscheck(False)
    def formMat(self, Vec X, Mat A, Mat I=None):
        cdef np.int64_t i, j, ia, ja, ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da4.globalToLocal(X,       self.localX)
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        xp = self.da4.getVecArray(self.localX)
        xh = self.da4.getVecArray(self.localXh)
        
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
        
        if I != None:
            I.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                
                # V_x
                row.index = (i,j, 0)
                row.field = 0
                
                # dt(dVx)
                # + fdudx(Vxp, dVx)
                # + fdudx(Vxh, dVx)
                # + fdudy(Vyp, dVx)
                # + fdudy(Vyh, dVx)
                # + udfdx(dVx, Vxp)
                # + udfdx(dVx, Vxh)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.fdudx(A_arr, Vxp, ix, jx, +1)
                self.fdudx(A_arr, Vxh, ix, jx, +1)
                self.fdudy(A_arr, Vyp, ix, jx, +1)
                self.fdudy(A_arr, Vyh, ix, jx, +1)
                self.udfdx(A_arr, Vxp, ix, jx, +1)
                self.udfdx(A_arr, Vxh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 0)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if I != None:
                    I.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # + udfdy(dVy, Vxp)
                # + udfdy(dVy, Vxh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdy(A_arr, Vxp, ix, jx, +1)
                self.udfdy(A_arr, Vxh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 1)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - fdudx(Bxp, dBx)
                # - fdudx(Bxh, dBx)
                # - fdudy(Byp, dBx)
                # - fdudy(Byh, dBx)
                # - udfdx(dBx, Bxp)
                # - udfdx(dBx, Bxh)
                
                A_arr = np.zeros((3,3))
                
                self.fdudx(A_arr, Bxp, ix, jx, -1)
                self.fdudx(A_arr, Bxh, ix, jx, -1)
                self.fdudy(A_arr, Byp, ix, jx, -1)
                self.fdudy(A_arr, Byh, ix, jx, -1)
                self.udfdx(A_arr, Bxp, ix, jx, -1)
                self.udfdx(A_arr, Bxh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 2)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - udfdy(dBy, Bxp)
                # - udfdy(dBy, Bxh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdy(A_arr, Bxp, ix, jx, -1)
                self.udfdy(A_arr, Bxh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 3)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # V_y
                row.index = (i,j, 1)
                row.field = 0
                
                # + udfdx(dVx, Vyp)
                # + udfdx(dVx, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdx(A_arr, Vyp, ix, jx, +1)
                self.udfdx(A_arr, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 0)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(Vy)
                # + fdudx(Vxp, Vy)
                # + fdudx(Vxh, Vy)
                # + fdudy(Vyp, Vy)
                # + fdudy(Vyh, Vy)
                # + udfdy(dVy, Vyp)
                # + udfdy(dVy, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.fdudx(A_arr, Vxp, ix, jx, +1)
                self.fdudx(A_arr, Vxh, ix, jx, +1)
                self.fdudy(A_arr, Vyp, ix, jx, +1)
                self.fdudy(A_arr, Vyh, ix, jx, +1)
                self.udfdy(A_arr, Vyp, ix, jx, +1)
                self.udfdy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 1)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if I != None:
                    I.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # - udfdx(dBx, Byp)
                # - udfdx(dBx, Byh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdx(A_arr, Byp, ix, jx, -1)
                self.udfdx(A_arr, Byh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 2)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - fdudx(Bxp, By)
                # - fdudx(Bxh, By)
                # - fdudy(Byp, By)
                # - fdudy(Byh, By)
                # - udfdy(dBy, Byp)
                # - udfdy(dBy, Byh)
                
                A_arr = np.zeros((3,3))
                
                self.fdudx(A_arr, Bxp, ix, jx, -1)
                self.fdudx(A_arr, Bxh, ix, jx, -1)
                self.fdudy(A_arr, Byp, ix, jx, -1)
                self.fdudy(A_arr, Byh, ix, jx, -1)
                self.udfdy(A_arr, Byp, ix, jx, -1)
                self.udfdy(A_arr, Byh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 3)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # B_x
                row.index = (i, j, 2)
                row.field = 0
                
                # - fdudx(Bxp, dVx)
                # - fdudx(Bxh, dVx)
                # - fdudy(Byp, dVx)
                # - fdudy(Byh, dVx)
                # + fdudx(dVx, Bxp)
                # + fdudx(dVx, Bxh)
                
                A_arr = np.zeros((3,3))
                
                self.fdudx(A_arr, Bxp, ix, jx, -1)
                self.fdudx(A_arr, Bxh, ix, jx, -1)
                self.fdudy(A_arr, Byp, ix, jx, -1)
                self.fdudy(A_arr, Byh, ix, jx, -1)
                self.udfdx(A_arr, Bxp, ix, jx, +1)
                self.udfdx(A_arr, Bxh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 0)
                        A.setValueStencil(row, col, A_arr[ia,ja])


                # + udfdy(dVy, Bxp)
                # + udfdy(dVy, Bxh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdy(A_arr, Bxp, ix, jx, +1)
                self.udfdy(A_arr, Bxh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 1)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(dBx)
                # + fdudx(Vxp, dBx)
                # + fdudx(Vxh, dBx)
                # + fdudy(Vyp, dBx)
                # + fdudy(Vyh, dBx)
                # - udfdx(dBx, Vxp)
                # - udfdx(dBx, Vxh)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.fdudx(A_arr, Vxp, ix, jx, +1)
                self.fdudx(A_arr, Vxh, ix, jx, +1)
                self.fdudy(A_arr, Vyp, ix, jx, +1)
                self.fdudy(A_arr, Vyh, ix, jx, +1)
                self.udfdx(A_arr, Vxp, ix, jx, -1)
                self.udfdx(A_arr, Vxh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 2)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if I != None:
                    I.setValueStencil(row, row, 1. / A_arr[1,1])
                
                # - udfdy(dBy, Vxp)
                # - udfdy(dBy, Vxh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdy(A_arr, Vxp, ix, jx, -1)
                self.udfdy(A_arr, Vxh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 3)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # B_y
                row.index = (i,j, 3)
                row.field = 0
                
                # + udfdx(dVx, Byp)
                # + udfdx(dVx, Byh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdx(A_arr, Byp, ix, jx, +1)
                self.udfdx(A_arr, Byh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 0)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - fdudx(Bxp, dVy)
                # - fdudx(Bxh, dVy)
                # - fdudy(Byp, dVy)
                # - fdudy(Byh, dVy)
                # + udfdy(dVy, Byp)
                # + udfdy(dVy, Byh)
                
                A_arr = np.zeros((3,3))
                
                self.fdudx(A_arr, Bxp, ix, jx, -1)
                self.fdudx(A_arr, Bxh, ix, jx, -1)
                self.fdudy(A_arr, Byp, ix, jx, -1)
                self.fdudy(A_arr, Byh, ix, jx, -1)
                self.udfdy(A_arr, Byp, ix, jx, +1)
                self.udfdy(A_arr, Byh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 1)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - udfdx(dBx, Vyp)
                # - udfdx(dBx, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdx(A_arr, Vyp, ix, jx, -1)
                self.udfdx(A_arr, Vyh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 2)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(dBy)
                # + fdudx(Vxp, dBy)
                # + fdudx(Vxh, dBy)
                # + fdudy(Vyp, dBy)
                # + fdudy(Vyh, dBy)
                # - udfdy(dBy, Vyp)
                # - udfdy(dBy, Vyh)

                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.fdudx(A_arr, Vxp, ix, jx, +1)
                self.fdudx(A_arr, Vxh, ix, jx, +1)
                self.fdudy(A_arr, Vyp, ix, jx, +1)
                self.fdudy(A_arr, Vyh, ix, jx, +1)
                self.udfdy(A_arr, Vyp, ix, jx, -1)
                self.udfdy(A_arr, Vyh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 3)
                        A.setValueStencil(row, col, A_arr[ia,ja])

                if I != None:
                    I.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
        A.assemble()
        
        if I != None:
            I.assemble()
        
#        if PETSc.COMM_WORLD.getRank() == 0:
#            print("     Matrix")
        
                
    
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
                        (-1, -1, - 1. * self.fac_gradx),
                        (+1, -1, + 1. * self.fac_gradx),
                        (-1,  0, - 2. * self.fac_gradx),
                        (+1,  0, + 2. * self.fac_gradx),
                        (-1, +1, - 1. * self.fac_gradx),
                        (+1, +1, + 1. * self.fac_gradx),
                    ]:
                    
                    tcol = ( self.nx * (col / self.nx) + ( col % self.nx + ti + self.nx) % self.nx + tj * self.nx + self.nx * self.ny) % (self.nx * self.ny)
                    B.setValue(row, tcol, value)
                
                
                # dy
                row = col + self.nx * self.ny
                
                for ti, tj, value in [
                        (-1, -1, - 1. * self.fac_grady),
                        (-1, +1, + 1. * self.fac_grady),
                        ( 0, -1, - 2. * self.fac_grady),
                        ( 0, +1, + 2. * self.fac_grady),
                        (+1, -1, - 1. * self.fac_grady),
                        (+1, +1, + 1. * self.fac_grady),
                    ]:
        
                    tcol = ( self.nx * (col / self.nx) + ( col % self.nx + ti + self.nx) % self.nx + tj * self.nx + self.nx * self.ny) % (self.nx * self.ny)
                    B.setValue(row, tcol, value)
        
        
        B.assemble()
        
        

    @cython.boundscheck(False)
    def formMatBT(self, Mat B not None):
        cdef np.int64_t i, j, ti, tj
        cdef np.int64_t row, col, tcol
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        B.zeroEntries()
        
        
        for i in np.arange(xs, xe):
            for j in np.arange(ys, ye):
                row = i + j * self.nx 
                
                # dx
#                for ti, tj, value in [
#                        (-1, -1, - 1. * self.fac_divx),
#                        (+1, -1, + 1. * self.fac_divx),
#                        (-1,  0, - 2. * self.fac_divx),
#                        (+1,  0, + 2. * self.fac_divx),
#                        (-1, +1, - 1. * self.fac_divx),
#                        (+1, +1, + 1. * self.fac_divx),
#                    ]:
                for ti, tj, value in [
                        (-1,  0, - self.fac_divx),
                        ( 0,  0, + self.fac_divx),
                    ]:
                    
                    tcol = ( self.nx * (row / self.nx) + ( row % self.nx + ti + self.nx) % self.nx + tj * self.nx + self.nx * self.ny) % (self.nx * self.ny)
                    B.setValue(row, tcol, value)
                
                
                # dy
#                for ti, tj, value in [
#                        (-1, -1, - 1. * self.fac_divy),
#                        (-1, +1, + 1. * self.fac_divy),
#                        ( 0, -1, - 2. * self.fac_divy),
#                        ( 0, +1, + 2. * self.fac_divy),
#                        (+1, -1, - 1. * self.fac_divy),
#                        (+1, +1, + 1. * self.fac_divy),
#                    ]:
                for ti, tj, value in [
                        ( 0, -1, - self.fac_divy),
                        ( 0,  0, + self.fac_divy),
                    ]:
        
                    
                    tcol = ( self.nx * (row / self.nx) + ( row % self.nx + ti + self.nx) % self.nx + tj * self.nx + self.nx * self.ny) % (self.nx * self.ny) \
                         + self.nx * self.ny
                    B.setValue(row, tcol, value)
        
        
        B.assemble()
        
        

    @cython.boundscheck(False)
    def formMatDx(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in np.arange(xs, xe):
            for j in np.arange(ys, ye):
                
                row.index = (i,j)
                row.field = 0
                
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_dx),
                        ((i+1, j-1), + 1. * self.fac_dx),
                        ((i-1, j  ), - 2. * self.fac_dx),
                        ((i+1, j  ), + 2. * self.fac_dx),
                        ((i-1, j+1), - 1. * self.fac_dx),
                        ((i+1, j+1), + 1. * self.fac_dx),
                    ]:
                                            
                    col.index = index
                    col.field = 0
                    A.setValueStencil(row, col, value)
        
        A.assemble()
        
        
    
    @cython.boundscheck(False)
    def formMatDy(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in np.arange(xs, xe):
            for j in np.arange(ys, ye):
                
                row.index = (i,j)
                row.field = 0
                
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_dy),
                        ((i-1, j+1), + 1. * self.fac_dy),
                        ((i,   j-1), - 2. * self.fac_dy),
                        ((i,   j+1), + 2. * self.fac_dy),
                        ((i+1, j-1), - 1. * self.fac_dy),
                        ((i+1, j+1), + 1. * self.fac_dy),
                    ]:
                                            
                    col.index = index
                    col.field = 0
                    A.setValueStencil(row, col, value)
        
        A.assemble()
        
        
    
    @cython.boundscheck(False)
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

    
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Pressure")



    @cython.boundscheck(False)
    def timestep(self, np.ndarray[np.float64_t, ndim=3] x,
                       np.ndarray[np.float64_t, ndim=2] p,
                       np.ndarray[np.float64_t, ndim=3] y):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
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
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] A,
                               np.uint64_t i, np.uint64_t j):
        
        # (i-1, j-1)
        A[0,0] += 1. * self.fac_dt
        
        # (i-1, j  )
        A[0,1] += 2. * self.fac_dt
        
        # (i-1, j+1)
        A[0,2] += 1. * self.fac_dt
        
        # (i,   j-1)
        A[1,0] += 2. * self.fac_dt
        
        # (i,   j  )
        A[1,1] += 4. * self.fac_dt
        
        # (i,   j+1)
        A[1,2] += 2. * self.fac_dt
        
        # (i+1, j-1)
        A[2,0] += 1. * self.fac_dt
        
        # (i+1, j  )
        A[2,1] += 2. * self.fac_dt
        
        # (i+1, j+1)
        A[2,2] += 1. * self.fac_dt
        
        

    @cython.boundscheck(False)
    cpdef np.float64_t fdudx(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign):
        
        # (i-1, j-1)
        A[0,0] += - 1. * ( F[i-1, j  ] + F[i-1, j-1] ) * self.fac_fdudx * sign \
                  - 1. * ( F[i,   j  ] + F[i,   j-1] ) * self.fac_fdudx * sign
        
        # (i-1, j  )
        A[0,1] += - 1. * ( F[i-1, j-1] + F[i,   j-1] ) * self.fac_fdudx * sign \
                  - 2. * ( F[i-1, j  ] + F[i,   j  ] ) * self.fac_fdudx * sign \
                  - 1. * ( F[i-1, j+1] + F[i,   j+1] ) * self.fac_fdudx * sign
        
        # (i-1, j+1)
        A[0,2] += - 1. * ( F[i-1, j  ] + F[i-1, j+1] ) * self.fac_fdudx * sign \
                  - 1. * ( F[i,   j  ] + F[i,   j+1] ) * self.fac_fdudx * sign
        
        # (i,   j-1)
        A[1,0] += - 1. * ( F[i+1, j-1] - F[i-1, j-1] ) * self.fac_fdudx * sign \
                  - 1. * ( F[i+1, j  ] - F[i-1, j  ] ) * self.fac_fdudx * sign
        
        # (i,   j  )
        A[1,1] += - 1. * ( F[i+1, j-1] - F[i-1, j-1] ) * self.fac_fdudx * sign \
                  - 2. * ( F[i+1, j  ] - F[i-1, j  ] ) * self.fac_fdudx * sign \
                  - 1. * ( F[i+1, j+1] - F[i-1, j+1] ) * self.fac_fdudx * sign
        
        # (i,   j+1)
        A[1,2] += - 1. * ( F[i+1, j  ] - F[i-1, j  ] ) * self.fac_fdudx * sign \
                  - 1. * ( F[i+1, j+1] - F[i-1, j+1] ) * self.fac_fdudx * sign
        
        # (i+1, j-1)
        A[2,0] += + 1. * ( F[i,   j-1] + F[i,   j  ] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j-1] + F[i+1, j  ] ) * self.fac_fdudx * sign
        
        # (i+1, j  )
        A[2,1] += + 1. * ( F[i,   j-1] + F[i+1, j-1] ) * self.fac_fdudx * sign \
                  + 2. * ( F[i,   j  ] + F[i+1, j  ] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i,   j+1] + F[i+1, j+1] ) * self.fac_fdudx * sign
        
        # (i+1, j+1)
        A[2,2] += + 1. * ( F[i,   j  ] + F[i,   j+1] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j  ] + F[i+1, j+1] ) * self.fac_fdudx * sign
        
        

    @cython.boundscheck(False)
    cpdef np.float64_t fdudy(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign):
        
        # (i-1, j-1)
#                     
#                     - 1 * F[i-1, j-1] * U[i-1, j-1] \
#                     - 1 * F[i-1, j  ] * U[i-1, j-1] \
#                     - 1 * F[i,   j-1] * U[i-1, j-1] \
#                     - 1 * F[i,   j  ] * U[i-1, j-1] \
#                     
        A[0,0] += - 1. * ( F[i-1, j-1] + F[i-1, j  ] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i,   j-1] + F[i,   j  ] ) * self.fac_fdudy * sign
        
        # (i-1, j  )
#                     
#                     - 1 * F[i-1, j+1] * U[i-1, j  ] \
#                     + 1 * F[i-1, j-1] * U[i-1, j  ] \
#                     - 1 * F[i,   j+1] * U[i-1, j  ] \
#                     + 1 * F[i,   j-1] * U[i-1, j  ] \
#                     
        A[0,1] += - 1. * ( F[i-1, j+1] - F[i-1, j-1] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i,   j+1] - F[i,   j-1] ) * self.fac_fdudy * sign
        
        # (i-1, j+1)
#                     
#                     + 1 * F[i-1, j  ] * U[i-1, j+1] \
#                     + 1 * F[i-1, j+1] * U[i-1, j+1] \
#                     + 1 * F[i,   j  ] * U[i-1, j+1] \
#                     + 1 * F[i,   j+1] * U[i-1, j+1] \
#                     
        A[0,2] += + 1. * ( F[i-1, j  ] + F[i-1, j+1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i,   j  ] + F[i,   j+1] ) * self.fac_fdudy * sign
        
        # (i,   j-1)
#                     
#                     - 1 * F[i-1, j-1] * U[i,   j-1] \
#                     - 1 * F[i-1, j  ] * U[i,   j-1] \
#                     - 2 * F[i,   j-1] * U[i,   j-1] \
#                     - 2 * F[i,   j  ] * U[i,   j-1] \
#                     - 1 * F[i+1, j-1] * U[i,   j-1] \
#                     - 1 * F[i+1, j  ] * U[i,   j-1] \
#                     
        A[1,0] += - 1. * ( F[i-1, j-1] + F[i-1, j  ] ) * self.fac_fdudy * sign \
                  - 2. * ( F[i,   j-1] + F[i,   j  ] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j-1] + F[i+1, j  ] ) * self.fac_fdudy * sign
        
        # (i,   j  )
#                     
#                     + 1 * F[i-1, j-1] * U[i,   j  ] \
#                     - 1 * F[i-1, j+1] * U[i,   j  ] \
#                     + 2 * F[i,   j-1] * U[i,   j  ] \
#                     - 2 * F[i,   j+1] * U[i,   j  ] \
#                     + 1 * F[i+1, j-1] * U[i,   j  ] \
#                     - 1 * F[i+1, j+1] * U[i,   j  ] \
#                     
        A[1,1] += - 1. * ( F[i-1, j+1] - F[i-1, j-1] ) * self.fac_fdudy * sign \
                  - 2. * ( F[i,   j+1] - F[i,   j-1] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j+1] - F[i+1, j-1] ) * self.fac_fdudy * sign
        
        # (i,   j+1)
#                     
#                     + 1 * F[i-1, j  ] * U[i,   j+1] \
#                     + 1 * F[i-1, j+1] * U[i,   j+1] \
#                     + 2 * F[i,   j  ] * U[i,   j+1] \
#                     + 2 * F[i,   j+1] * U[i,   j+1] \
#                     + 1 * F[i+1, j  ] * U[i,   j+1] \
#                     + 1 * F[i+1, j+1] * U[i,   j+1] \
#                     
        A[1,2] += + 1. * ( F[i-1, j  ] + F[i-1, j+1] ) * self.fac_fdudy * sign \
                  + 2. * ( F[i,   j  ] + F[i,   j+1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j  ] + F[i+1, j+1] ) * self.fac_fdudy * sign
        
        # (i+1, j-1)
#
#                     - 1 * F[i,   j-1] * U[i+1, j-1] \
#                     - 1 * F[i,   j  ] * U[i+1, j-1] \
#                     - 1 * F[i+1, j-1] * U[i+1, j-1] \
#                     - 1 * F[i+1, j  ] * U[i+1, j-1] \
#                     
        A[2,0] += - 1. * ( F[i,   j-1] + F[i,   j  ] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j-1] + F[i+1, j  ] ) * self.fac_fdudy * sign
        
        # (i+1, j  )
#
#                     + 1 * F[i,   j-1] * U[i+1, j  ] \
#                     - 1 * F[i,   j+1] * U[i+1, j  ] \
#                     + 1 * F[i+1, j-1] * U[i+1, j  ] \
#                     - 1 * F[i+1, j+1] * U[i+1, j  ] \
#                     
        A[2,1] += - 1. * ( F[i,   j+1] - F[i,   j-1] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j+1] - F[i+1, j-1] ) * self.fac_fdudy * sign
        
        # (i+1, j+1)
#
#                     + 1 * F[i,   j  ] * U[i+1, j+1] \
#                     + 1 * F[i,   j+1] * U[i+1, j+1] \
#                     + 1 * F[i+1, j  ] * U[i+1, j+1] \
#                     + 1 * F[i+1, j+1] * U[i+1, j+1] \
#                     
        A[2,2] += + 1. * ( F[i,   j  ] + F[i,   j+1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j  ] + F[i+1, j+1] ) * self.fac_fdudy * sign
        

        
    @cython.boundscheck(False)
    cpdef np.float64_t udfdx(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign):
        
        # (i-1, j-1)
        A[0,0] += + 1. * ( F[i,   j-1] - F[i-1, j-1] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i,   j  ] - F[i-1, j  ] ) * self.fac_fdudx * sign
        
        # (i-1, j  )
        A[0,1] += + 1. * ( F[i,   j-1] - F[i-1, j-1] ) * self.fac_fdudx * sign \
                  + 2. * ( F[i,   j  ] - F[i-1, j  ] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i,   j+1] - F[i-1, j+1] ) * self.fac_fdudx * sign
        
        # (i-1, j+1)
        A[0,2] += + 1. * ( F[i,   j+1] - F[i-1, j+1] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i,   j  ] - F[i-1, j  ] ) * self.fac_fdudx * sign
        
        # (i,   j-1)
        A[1,0] += + 1. * ( F[i+1, j-1] - F[i-1, j-1] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j  ] - F[i-1, j  ] ) * self.fac_fdudx * sign
        
        # (i,   j  )
        A[1,1] += + 1. * ( F[i+1, j-1] - F[i-1, j-1] ) * self.fac_fdudx * sign \
                  + 2. * ( F[i+1, j  ] - F[i-1, j  ] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j+1] - F[i-1, j+1] ) * self.fac_fdudx * sign
        
        # (i,   j+1)
        A[1,2] += + 1. * ( F[i+1, j  ] - F[i-1, j  ] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j+1] - F[i-1, j+1] ) * self.fac_fdudx * sign
        
        # (i+1, j-1)
        A[2,0] += + 1. * ( F[i+1, j-1] - F[i,   j-1] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j  ] - F[i,   j  ] ) * self.fac_fdudx * sign
        
        # (i+1, j  )
        A[2,1] += + 1. * ( F[i+1, j-1] - F[i,   j-1] ) * self.fac_fdudx * sign \
                  + 2. * ( F[i+1, j  ] - F[i,   j  ] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j+1] - F[i,   j+1] ) * self.fac_fdudx * sign
        
        # (i+1, j+1)
        A[2,2] += + 1. * ( F[i+1, j  ] - F[i,   j  ] ) * self.fac_fdudx * sign \
                  + 1. * ( F[i+1, j+1] - F[i,   j+1] ) * self.fac_fdudx * sign
        
    
    @cython.boundscheck(False)
    cpdef np.float64_t udfdy(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign):
        
        # (i-1, j-1)
        A[0,0] += + 1. * ( F[i-1, j  ] - F[i-1, j-1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i,   j  ] - F[i,   j-1] ) * self.fac_fdudy * sign
        
        # (i-1, j  )
        A[0,1] += + 1. * ( F[i-1, j+1] - F[i-1, j-1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i,   j+1] - F[i,   j-1] ) * self.fac_fdudy * sign
        
        # (i-1, j+1)
        A[0,2] += + 1. * ( F[i-1, j+1] - F[i-1, j  ] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i,   j+1] - F[i,   j  ] ) * self.fac_fdudy * sign
        
        # (i,   j-1)
        A[1,0] += + 1. * ( F[i-1, j  ] - F[i-1, j-1] ) * self.fac_fdudy * sign \
                  + 2. * ( F[i,   j  ] - F[i,   j-1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j  ] - F[i+1, j-1] ) * self.fac_fdudy * sign
        
        # (i,   j  )
        A[1,1] += + 1. * ( F[i-1, j+1] - F[i-1, j-1] ) * self.fac_fdudy * sign \
                  + 2. * ( F[i,   j+1] - F[i,   j-1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j+1] - F[i+1, j-1] ) * self.fac_fdudy * sign
        
        # (i,   j+1)
        A[1,2] += + 1. * ( F[i-1, j+1] - F[i-1, j  ] ) * self.fac_fdudy * sign \
                  + 2. * ( F[i,   j+1] - F[i,   j  ] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j+1] - F[i+1, j  ] ) * self.fac_fdudy * sign
        
        # (i+1, j-1)
        A[2,0] += + 1. * ( F[i,   j  ] - F[i,   j-1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j  ] - F[i+1, j-1] ) * self.fac_fdudy * sign
        
        # (i+1, j  )
        A[2,1] += + 1. * ( F[i,   j+1] - F[i,   j-1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j+1] - F[i+1, j-1] ) * self.fac_fdudy * sign
        
        # (i+1, j+1)
        A[2,2] += + 1. * ( F[i,   j+1] - F[i,   j  ] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j+1] - F[i+1, j  ] ) * self.fac_fdudy * sign
        
