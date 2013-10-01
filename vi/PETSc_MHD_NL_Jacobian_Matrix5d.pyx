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
    
    def __init__(self, DA da1, DA da5,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        # distributed array
        self.da1 = da1
        self.da5 = da5
        
        # grid size
        self.nx = nx
        self.ny = ny
        
        # step size
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        # factors of derivatives
#        self.fac_dx    = 1.0 / 4. / 2. / self.hx
#        self.fac_dy    = 1.0 / 4. / 2. / self.hy
        self.fac_dx    = 0.5 / 4. / 2. / self.hx
        self.fac_dy    = 0.5 / 4. / 2. / self.hy
        self.fac_dudt  = 1.   / 16. / self.ht
        self.fac_fdudx = 0.25 / 32. / self.hx
        self.fac_fdudy = 0.25 / 32. / self.hy
        
        
        # create history vectors
        self.Xh = self.da5.createGlobalVec()
        self.Xp = self.da5.createGlobalVec()
        
        # create local vectors
        self.localXh = da5.createLocalVec()
        self.localXp = da5.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A, Mat P = None):
        cdef np.int64_t i, j, ia, ja, ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da5.globalToLocal(self.Xh, self.localXh)
        self.da5.globalToLocal(self.Xp, self.localXp)
        
        xh = self.da5.getVecArray(self.localXh)
        xp = self.da5.getVecArray(self.localXp)
        
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
                row.index = (i,j, 0)
                row.field = 0
                
                # dt(dVx)
                
                # + Vx dx Vx
                # + fdudx(Vxp, dVx)
                # + fdudx(Vxh, dVx)
                
                # + Vy dy Vx
                # + fdudy(Vyp, dVx)
                # + fdudy(Vyh, dVx)
                
                # + Vx dx Vx
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
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # + Vy dy Vx
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
                
                
                # - Bx dx Bx
                # - fdudx(Bxp, dBx)
                # - fdudx(Bxh, dBx)
                
                # - By dy Bx
                # - fdudy(Byp, dBx)
                # - fdudy(Byh, dBx)
                
                # - Bx dx Bx
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
                
                
                # - By dy Bx
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
                
                
                # dx(P)
                
                col.field = 0
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_dx),
                        ((i+1, j-1), + 1. * self.fac_dx),
                        ((i-1, j  ), - 2. * self.fac_dx),
                        ((i+1, j  ), + 2. * self.fac_dx),
                        ((i-1, j+1), - 1. * self.fac_dx),
                        ((i+1, j+1), + 1. * self.fac_dx),
                    ]:
                                            
                    col.index = (index[0], index[1], 4)
                    A.setValueStencil(row, col, value)
                
                
                # V_y
                row.index = (i,j, 1)
                row.field = 0
                
                # + Vx dx Vy
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
                
                # + Vx dx Vy
                # + fdudx(Vxp, dVy)
                # + fdudx(Vxh, dVy)
                
                # + Vy dy Vy
                # + fdudy(Vyp, dVy)
                # + fdudy(Vyh, dVy)
                
                # + Vy dy Vy
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
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # - Bx dx By
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
                
                
                # - Bx dx By
                # - fdudx(Bxp, dBy)
                # - fdudx(Bxh, dBy)
                
                # - By dy By
                # - fdudy(Byp, dBy)
                # - fdudy(Byh, dBy)
                
                # - By dy By
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
                
                
                # dy(P)
                
                col.field = 0
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_dy),
                        ((i-1, j+1), + 1. * self.fac_dy),
                        ((i,   j-1), - 2. * self.fac_dy),
                        ((i,   j+1), + 2. * self.fac_dy),
                        ((i+1, j-1), - 1. * self.fac_dy),
                        ((i+1, j+1), + 1. * self.fac_dy),
                    ]:
                                            
                    col.index = (index[0], index[1], 4)
                    A.setValueStencil(row, col, value)
                
                
                # B_x
                row.index = (i,j, 2)
                row.field = 0
                
                # - Bx dx Vx
                # - fdudx(Bxp, dVx)
                # - fdudx(Bxh, dVx)
                
                # - By dy Vx
                # - fdudy(Byp, dVx)
                # - fdudy(Byh, dVx)
                
                # + Vx dx Bx
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


                # + Vy dy Bx
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
                
                # + Vx dx Bx
                # + fdudx(Vxp, dBx)
                # + fdudx(Vxh, dBx)

                # + Vy dy Bx
                # + fdudy(Vyp, dBx)
                # + fdudy(Vyh, dBx)
                
                # - Bx dx Vx
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
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # - By dy Vx
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
                
                # Vx dx By
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
                
                
                # Bx dx Vy
                # - fdudx(Bxp, dVy)
                # - fdudx(Bxh, dVy)
                
                # By dy Vy
                # - fdudy(Byp, dVy)
                # - fdudy(Byh, dVy)
                
                # Vy dy By
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
                
                
                # Bx dx Vy
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

                # + Vx dx By
                # + fdudx(Vxp, dBy)
                # + fdudx(Vxh, dBy)
                
                # + Vy dy By
                # + fdudy(Vyp, dBy)
                # + fdudy(Vyh, dBy)
                
                # - By dy Vy
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

                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # P
                row.index = (i,j, 4)
                row.field = 0
                
                # dx(Vx)
                
                col.field = 0
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_dx),
                        ((i+1, j-1), + 1. * self.fac_dx),
                        ((i-1, j  ), - 2. * self.fac_dx),
                        ((i+1, j  ), + 2. * self.fac_dx),
                        ((i-1, j+1), - 1. * self.fac_dx),
                        ((i+1, j+1), + 1. * self.fac_dx),
                    ]:
                                            
                    col.index = (index[0], index[1], 0)
                    A.setValueStencil(row, col, value)
                
                
                # dy(Vy)
                
                col.field = 0
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_dy),
                        ((i-1, j+1), + 1. * self.fac_dy),
                        ((i,   j-1), - 2. * self.fac_dy),
                        ((i,   j+1), + 2. * self.fac_dy),
                        ((i+1, j-1), - 1. * self.fac_dy),
                        ((i+1, j+1), + 1. * self.fac_dy),
                    ]:
                                            
                    col.index = (index[0], index[1], 1)
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
        if P != None:
            P.assemble()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Matrix")
        
                
    

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
    cdef np.float64_t fdudy(self, np.ndarray[np.float64_t, ndim=2] A,
                                  np.ndarray[np.float64_t, ndim=2] F,
                                  np.uint64_t i, np.uint64_t j,
                                  np.float64_t sign):
        
        # (i-1, j-1)
        A[0,0] += - 1. * ( F[i-1, j-1] + F[i-1, j  ] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i,   j-1] + F[i,   j  ] ) * self.fac_fdudy * sign
        
        # (i-1, j  )
        A[0,1] += - 1. * ( F[i-1, j+1] - F[i-1, j-1] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i,   j+1] - F[i,   j-1] ) * self.fac_fdudy * sign
        
        # (i-1, j+1)
        A[0,2] += + 1. * ( F[i-1, j  ] + F[i-1, j+1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i,   j  ] + F[i,   j+1] ) * self.fac_fdudy * sign
        
        # (i,   j-1)
        A[1,0] += - 1. * ( F[i-1, j-1] + F[i-1, j  ] ) * self.fac_fdudy * sign \
                  - 2. * ( F[i,   j-1] + F[i,   j  ] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j-1] + F[i+1, j  ] ) * self.fac_fdudy * sign
        
        # (i,   j  )
        A[1,1] += - 1. * ( F[i-1, j+1] - F[i-1, j-1] ) * self.fac_fdudy * sign \
                  - 2. * ( F[i,   j+1] - F[i,   j-1] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j+1] - F[i+1, j-1] ) * self.fac_fdudy * sign
        
        # (i,   j+1)
        A[1,2] += + 1. * ( F[i-1, j  ] + F[i-1, j+1] ) * self.fac_fdudy * sign \
                  + 2. * ( F[i,   j  ] + F[i,   j+1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j  ] + F[i+1, j+1] ) * self.fac_fdudy * sign
        
        # (i+1, j-1)
        A[2,0] += - 1. * ( F[i,   j-1] + F[i,   j  ] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j-1] + F[i+1, j  ] ) * self.fac_fdudy * sign
        
        # (i+1, j  )
        A[2,1] += - 1. * ( F[i,   j+1] - F[i,   j-1] ) * self.fac_fdudy * sign \
                  - 1. * ( F[i+1, j+1] - F[i+1, j-1] ) * self.fac_fdudy * sign
        
        # (i+1, j+1)
        A[2,2] += + 1. * ( F[i,   j  ] + F[i,   j+1] ) * self.fac_fdudy * sign \
                  + 1. * ( F[i+1, j  ] + F[i+1, j+1] ) * self.fac_fdudy * sign
        
        
    @cython.boundscheck(False)
    cdef np.float64_t udfdx(self, np.ndarray[np.float64_t, ndim=2] A,
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
    cdef np.float64_t udfdy(self, np.ndarray[np.float64_t, ndim=2] A,
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
        
