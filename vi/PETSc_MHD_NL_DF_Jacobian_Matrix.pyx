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
        self.fac_dt    = 1.   / 16. / self.ht
        self.fac_dx    = 0.25 / 16. / self.hx
        self.fac_dy    = 0.25 / 16. / self.hy
        self.fac_gradx = 0.5  / 8.  / self.hx
        self.fac_grady = 0.5  / 8.  / self.hy
        
        
        # create history vectors
        self.Xh = self.da5.createGlobalVec()
#        self.Xp = self.da5.createGlobalVec()
        
        # create local vectors
        self.localXh = da5.createLocalVec()
        self.localXp = da5.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
#    def update_previous(self, Vec X):
#        X.copy(self.Xp)
        
    
    @cython.boundscheck(False)
    def formMat(self, Vec X, Mat A, Mat P = None):
        cdef np.int64_t i, j, ia, ja, ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da5.globalToLocal(X,       self.localXp)
        self.da5.globalToLocal(self.Xh, self.localXh)
        
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
                # + dx(Vxp, dVx)
                # + dx(Vxh, dVx)
                # + dx(dVx, Vxp)
                # + dx(dVx, Vxh)
                # + dy(dVx, Vyp)
                # + dy(dVx, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.dx(A_arr, Vxp, ix, jx, +2)
                self.dx(A_arr, Vxh, ix, jx, +2)
                self.dy(A_arr, Vyp, ix, jx, +1)
                self.dy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 0)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # + dy(Vxp, dVy)
                # + dy(Vxh, dVy)
                
                A_arr = np.zeros((3,3))
                
                self.dy(A_arr, Vxp, ix, jx, +1)
                self.dy(A_arr, Vxh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 1)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - dx(Bxp, dBx)
                # - dx(Bxh, dBx)
                # - dx(dBx, Bxp)
                # - dx(dBx, Bxh)
                # - dy(dBx, Byp)
                # - dy(dBx, Byh)
                
                A_arr = np.zeros((3,3))
                
                self.dx(A_arr, Bxp, ix, jx, -2)
                self.dx(A_arr, Bxh, ix, jx, -2)
                self.dy(A_arr, Byp, ix, jx, -1)
                self.dy(A_arr, Byh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 2)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                

                # - dy(Bxp, dBy)
                # - dy(Bxh, dBy)
                
                A_arr = np.zeros((3,3))
                
                self.dy(A_arr, Bxp, ix, jx, -1)
                self.dy(A_arr, Bxh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 3)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dx(P)
                
                col.field = 0
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_gradx),
                        ((i+1, j-1), + 1. * self.fac_gradx),
                        ((i-1, j  ), - 2. * self.fac_gradx),
                        ((i+1, j  ), + 2. * self.fac_gradx),
                        ((i-1, j+1), - 1. * self.fac_gradx),
                        ((i+1, j+1), + 1. * self.fac_gradx),
                    ]:
                                            
                    col.index = (index[0], index[1], 4)
                    A.setValueStencil(row, col, value)
                
                
                # V_y
                row.index = (i,j, 1)
                row.field = 0
                
                # + dx(dVx, Vyp)
                # + dx(dVx, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.dx(A_arr, Vyp, ix, jx, +1)
                self.dx(A_arr, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 0)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(Vy)
                # + dx(Vxp, dVy)
                # + dx(Vxh, dVy)
                # + dy(Vyp, dVy)
                # + dy(Vyh, dVy)
                # + dy(dVy, Vyp)
                # + dy(dVy, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.dx(A_arr, Vxp, ix, jx, +1)
                self.dx(A_arr, Vxh, ix, jx, +1)
                self.dy(A_arr, Vyp, ix, jx, +2)
                self.dy(A_arr, Vyh, ix, jx, +2)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 1)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # - dx(dBx, Byp)
                # - dx(dBx, Byh)
                
                A_arr = np.zeros((3,3))
                
                self.dx(A_arr, Byp, ix, jx, -1)
                self.dx(A_arr, Byh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 2)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - dx(Bxp, dBy)
                # - dx(Bxh, dBy)
                # - dy(Byp, dBy)
                # - dy(Byh, dBy)
                # - dy(dBy, Byp)
                # - dy(dBy, Byh)
                
                A_arr = np.zeros((3,3))
                
                self.dx(A_arr, Bxp, ix, jx, -1)
                self.dx(A_arr, Bxh, ix, jx, -1)
                self.dy(A_arr, Byp, ix, jx, -2)
                self.dy(A_arr, Byh, ix, jx, -2)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 3)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dy(P)
                
                col.field = 0
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_grady),
                        ((i-1, j+1), + 1. * self.fac_grady),
                        ((i,   j-1), - 2. * self.fac_grady),
                        ((i,   j+1), + 2. * self.fac_grady),
                        ((i+1, j-1), - 1. * self.fac_grady),
                        ((i+1, j+1), + 1. * self.fac_grady),
                    ]:
                                            
                    col.index = (index[0], index[1], 4)
                    A.setValueStencil(row, col, value)
                
                
                # B_x
                row.index = (i,j, 2)
                row.field = 0
                
                # - dy(Byp, dVx)
                # - dy(Byh, dVx)
                
                A_arr = np.zeros((3,3))
                
                self.dy(A_arr, Byp, ix, jx, -1)
                self.dy(A_arr, Byh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 0)
                        A.setValueStencil(row, col, A_arr[ia,ja])


                # + dy(Bxp, dVy)
                # + dy(Bxh, dVy)
                
                A_arr = np.zeros((3,3))
                
                self.dy(A_arr, Bxp, ix, jx, +1)
                self.dy(A_arr, Bxh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 1)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(dBx)
                # + dy(dBx, Vyp)
                # + dy(dBx, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.dy(A_arr, Vyp, ix, jx, +1)
                self.dy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 2)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # - dy(dBy, Vxp)
                # - dy(dBy, Vxh)
                
                A_arr = np.zeros((3,3))
                
                self.dy(A_arr, Vxp, ix, jx, -1)
                self.dy(A_arr, Vxh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 3)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # B_y
                row.index = (i,j, 3)
                row.field = 0
                
                # + dx(Byp, dVx)
                # + dx(Byh, dVx)
                
                A_arr = np.zeros((3,3))
                
                self.dx(A_arr, Byp, ix, jx, +1)
                self.dx(A_arr, Byh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 0)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - dx(Bxp, dVy)
                # - dx(Bxh, dVy)
                
                A_arr = np.zeros((3,3))
                
                self.dx(A_arr, Bxp, ix, jx, -1)
                self.dx(A_arr, Bxh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 1)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - dx(dBx, Vyp)
                # - dx(dBx, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.dx(A_arr, Vyp, ix, jx, -1)
                self.dx(A_arr, Vyh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja, 2)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(dBy)
                # + dx(dBy, Vxp)
                # + dx(dBy, Vxh)

                A_arr = np.zeros((3,3))
                
                self.dt(A_arr, ix, jx)
                self.dx(A_arr, Vxp, ix, jx, +1)
                self.dx(A_arr, Vxh, ix, jx, +1)
                
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
                        ((i-1, j-1), - 1. * self.fac_gradx),
                        ((i+1, j-1), + 1. * self.fac_gradx),
                        ((i-1, j  ), - 2. * self.fac_gradx),
                        ((i+1, j  ), + 2. * self.fac_gradx),
                        ((i-1, j+1), - 1. * self.fac_gradx),
                        ((i+1, j+1), + 1. * self.fac_gradx),
                    ]:
                                            
                    col.index = (index[0], index[1], 0)
                    A.setValueStencil(row, col, value)
                
                
                # dy(Vy)
                
                col.field = 0
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_grady),
                        ((i-1, j+1), + 1. * self.fac_grady),
                        ((i,   j-1), - 2. * self.fac_grady),
                        ((i,   j+1), + 2. * self.fac_grady),
                        ((i+1, j-1), - 1. * self.fac_grady),
                        ((i+1, j+1), + 1. * self.fac_grady),
                    ]:
                                            
                    col.index = (index[0], index[1], 1)
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
        if P != None:
            P.assemble()
        
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
                    
                    tcol = ( self.nx * (col / self.nx) + ( col % self.nx + ti + self.nx) % self.nx + tj * self.nx + self.nx * self.ny) % (self.nx * self.ny)
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
                for ti, tj, value in [
                        (-1, -1, - 1. * self.fac_dx),
                        (+1, -1, + 1. * self.fac_dx),
                        (-1,  0, - 2. * self.fac_dx),
                        (+1,  0, + 2. * self.fac_dx),
                        (-1, +1, - 1. * self.fac_dx),
                        (+1, +1, + 1. * self.fac_dx),
                    ]:
                    
                    tcol = ( self.nx * (row / self.nx) + ( row % self.nx + ti + self.nx) % self.nx + tj * self.nx + self.nx * self.ny) % (self.nx * self.ny)
                    B.setValue(row, tcol, value)
                
                
                # dy
                for ti, tj, value in [
                        (-1, -1, - 1. * self.fac_dy),
                        (-1, +1, + 1. * self.fac_dy),
                        ( 0, -1, - 2. * self.fac_dy),
                        ( 0, +1, + 2. * self.fac_dy),
                        (+1, -1, - 1. * self.fac_dy),
                        (+1, +1, + 1. * self.fac_dy),
                    ]:
        
                    
                    tcol = ( self.nx * (row / self.nx) + ( row % self.nx + ti + self.nx) % self.nx + tj * self.nx + self.nx * self.ny) % (self.nx * self.ny) \
                         + self.nx * self.ny
                    B.setValue(row, tcol, value)
        
        
        B.assemble()
        
        

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
    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] A,
                               np.ndarray[np.float64_t, ndim=2] F,
                               np.uint64_t i, np.uint64_t j,
                               np.float64_t sign):
        
        # (i-1, j-1)
        A[0,0] += sign * self.fac_dx * ( F[i-1, j-1] + F[i-1, j  ] )
        
        # (i-1, j  )
        A[0,1] += sign * self.fac_dx * ( F[i-1, j-1] + 2. * F[i-1, j  ] + F[i-1, j+1] )
        
        # (i-1, j+1)
        A[0,2] += sign * self.fac_dx * ( F[i-1, j  ] + F[i-1, j+1] )
        
        # (i+1, j-1)
        A[2,0] -= sign * self.fac_dx * ( F[i+1, j-1] + F[i+1, j  ] )
        
        # (i+1, j  )
        A[2,1] -= sign * self.fac_dx * ( F[i+1, j-1] + 2. * F[i+1, j  ] + F[i+1, j+1] )
        
        # (i+1, j+1)
        A[2,2] -= sign * self.fac_dx * ( F[i+1, j  ] + F[i+1, j+1] )
        
        

    @cython.boundscheck(False)
    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] A,
                               np.ndarray[np.float64_t, ndim=2] F,
                               np.uint64_t i, np.uint64_t j,
                               np.float64_t sign):
        
        # (i-1, j-1)
        A[0,0] += sign * self.fac_dy * ( F[i-1, j-1] + F[i,   j-1] ) 
        
        # (i-1, j+1)
        A[0,2] -= sign * self.fac_dy * ( F[i-1, j+1] + F[i,   j+1] ) 
        
        # (i,   j-1)
        A[1,0] += sign * self.fac_dy * ( F[i-1, j-1] + 2. * F[i,   j-1] + F[i+1, j-1] ) 
        
        # (i,   j+1)
        A[1,2] -= sign * self.fac_dy * ( F[i-1, j+1] + 2. * F[i,   j+1] + F[i+1, j+1] ) 
        
        # (i+1, j-1)
        A[2,0] += sign * self.fac_dy * ( F[i,   j-1] + F[i+1, j-1] ) 
        
        # (i+1, j+1)
        A[2,2] -= sign * self.fac_dy * ( F[i,   j+1] + F[i+1, j+1] ) 
        
