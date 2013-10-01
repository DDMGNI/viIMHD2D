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
                 np.float64_t ht, np.float64_t hx, np.float64_t hy,
                 eps=0.):
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
        self.fac_divx  = 1.0 / self.hx
        self.fac_divy  = 1.0 / self.hy
#        self.fac_divx  = 1.0  / 2. / self.hx
#        self.fac_divy  = 1.0  / 2. / self.hy
#        self.fac_divx  = 1.0  / 4. / 2. / self.hx
#        self.fac_divy  = 1.0  / 4. / 2. / self.hy
        
#        self.fac_grdx    = 1.0  / 4. / self.hx
#        self.fac_grdy    = 1.0  / 4. / self.hy
#        self.fac_grdx    = 0.5 / 4. / self.hx
#        self.fac_grdy    = 0.5 / 4. / self.hy
#        self.fac_grdx    = 1.0  / 4. / 2. / self.hx
#        self.fac_grdy    = 1.0  / 4. / 2. / self.hy
        self.fac_grdx    = 0.5  / 4. / 2. / self.hx
        self.fac_grdy    = 0.5  / 4. / 2. / self.hy
        
        self.fac_lapx  = 0.
        self.fac_lapy  = 0.
#        self.fac_lapx  = - eps  / 4. / self.hx**2
#        self.fac_lapy  = - eps  / 4. / self.hy**2
#        self.fac_lapx  = - eps / self.hx**2
#        self.fac_lapy  = - eps / self.hy**2
        
        self.fac_dt    = 1.   / 16. / self.ht
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
            ix = i-xs+2
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                
                # V_x
                row.index = (i,j)
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
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # + udfdy(dVy, Vxp)
                # + udfdy(dVy, Vxh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdy(A_arr, Vxp, ix, jx, +1)
                self.udfdy(A_arr, Vxh, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
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
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - udfdy(dBy, Bxp)
                # - udfdy(dBy, Bxh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdy(A_arr, Bxp, ix, jx, -1)
                self.udfdy(A_arr, Bxh, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dx(P)
                
                col.field = 4
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_grdx),
                        ((i+1, j-1), + 1. * self.fac_grdx),
                        ((i-1, j  ), - 2. * self.fac_grdx),
                        ((i+1, j  ), + 2. * self.fac_grdx),
                        ((i-1, j+1), - 1. * self.fac_grdx),
                        ((i+1, j+1), + 1. * self.fac_grdx),
                    ]:
#                for index, value in [
#                        ((i-1, j  ), - 1. * self.fac_grdx),
#                        ((i+1, j  ), + 1. * self.fac_grdx),
#                        ((i-1, j+1), - 1. * self.fac_grdx),
#                        ((i+1, j+1), + 1. * self.fac_grdx),
#                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # V_y
                row.index = (i,j)
                row.field = 1
                
                # + udfdx(dVx, Vyp)
                # + udfdx(dVx, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdx(A_arr, Vyp, ix, jx, +1)
                self.udfdx(A_arr, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
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
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # - udfdx(dBx, Byp)
                # - udfdx(dBx, Byh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdx(A_arr, Byp, ix, jx, -1)
                self.udfdx(A_arr, Byh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
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
                
                col.field = 3
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dy(P)
                col.field = 4
                for index, value in [
                        ((i-1, j-1), - 1. * self.fac_grdy),
                        ((i-1, j+1), + 1. * self.fac_grdy),
                        ((i,   j-1), - 2. * self.fac_grdy),
                        ((i,   j+1), + 2. * self.fac_grdy),
                        ((i+1, j-1), - 1. * self.fac_grdy),
                        ((i+1, j+1), + 1. * self.fac_grdy),
                    ]:
#                for index, value in [
#                        ((i,   j-1), - 1. * self.fac_grdy),
#                        ((i,   j+1), + 1. * self.fac_grdy),
#                        ((i+1, j-1), - 1. * self.fac_grdy),
#                        ((i+1, j+1), + 1. * self.fac_grdy),
#                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # B_x
                row.index = (i,j)
                row.field = 2
                
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
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])


                # + udfdy(dVy, Bxp)
                # + udfdy(dVy, Bxh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdy(A_arr, Bxp, ix, jx, +1)
                self.udfdy(A_arr, Bxh, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
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
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                # - udfdy(dBy, Vxp)
                # - udfdy(dBy, Vxh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdy(A_arr, Vxp, ix, jx, -1)
                self.udfdy(A_arr, Vxh, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # B_y
                row.index = (i,j)
                row.field = 3
                
                # + udfdx(dVx, Byp)
                # + udfdx(dVx, Byh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdx(A_arr, Byp, ix, jx, +1)
                self.udfdx(A_arr, Byh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
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
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - udfdx(dBx, Vyp)
                # - udfdx(dBx, Vyh)
                
                A_arr = np.zeros((3,3))
                
                self.udfdx(A_arr, Vyp, ix, jx, -1)
                self.udfdx(A_arr, Vyh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
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
                
                col.field = 3
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])

                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # P
                row.index = (i,j)
                row.field = 4
                
                # dx(Vx)
                
                col.field = 0
#                for index, value in [
#                        ((i-1, j-1), - 1.0 * self.fac_divx),
#                        ((i+1, j-1), + 1.0 * self.fac_divx),
#                        ((i-1, j  ), - 2.0 * self.fac_divx),
#                        ((i+1, j  ), + 2.0 * self.fac_divx),
#                        ((i-1, j+1), - 1.0 * self.fac_divx),
#                        ((i+1, j+1), + 1.0 * self.fac_divx),
#                    ]:
                    
#                for index, value in [
#                        ((i-1, j  ), - self.fac_divx),
#                        ((i+1, j  ), + self.fac_divx),
#                    ]:
                    
                for index, value in [
                        ((i-1, j  ), - self.fac_divx),
                        ((i,   j  ), + self.fac_divx),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # dy(Vy)
                
                col.field = 1
#                for index, value in [
#                        ((i-1, j-1), - 1.0 * self.fac_divy),
#                        ((i-1, j+1), + 1.0 * self.fac_divy),
#                        ((i,   j-1), - 2.0 * self.fac_divy),
#                        ((i,   j+1), + 2.0 * self.fac_divy),
#                        ((i+1, j-1), - 1.0 * self.fac_divy),
#                        ((i+1, j+1), + 1.0 * self.fac_divy),
#                    ]:
                    
#                for index, value in [
#                        ((i,   j-1), - self.fac_divy),
#                        ((i,   j+1), + self.fac_divy),
#                    ]:
                    
                for index, value in [
                        ((i, j-1), - self.fac_divy),
                        ((i, j  ), + self.fac_divy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # laplace(P)
                
                col.field = 4
                for index, value in [
                        ((i-1, j-1), + 1.0 * self.fac_lapx + 1.0 * self.fac_lapy),
                        ((i,   j-1), - 2.0 * self.fac_lapx + 2.0 * self.fac_lapy),
                        ((i+1, j-1), + 1.0 * self.fac_lapx + 1.0 * self.fac_lapy),
                        ((i-1, j  ), + 2.0 * self.fac_lapx - 2.0 * self.fac_lapy),
                        ((i,   j  ), - 4.0 * self.fac_lapx - 4.0 * self.fac_lapy),
                        ((i+1, j  ), + 2.0 * self.fac_lapx - 2.0 * self.fac_lapy),
                        ((i-1, j+1), + 1.0 * self.fac_lapx + 1.0 * self.fac_lapy),
                        ((i,   j+1), - 2.0 * self.fac_lapx + 2.0 * self.fac_lapy),
                        ((i+1, j+1), + 1.0 * self.fac_lapx + 1.0 * self.fac_lapy),
                    ]:
                    
#                for index, value in [
#                        ((i-1, j  ), + 1.0 * self.fac_lapx),
#                        ((i+1, j  ), + 1.0 * self.fac_lapx),
#                        ((i,   j  ), - 2.0 * self.fac_lapx - 2.0 * self.fac_lapy),
#                        ((i,   j-1), + 1.0 * self.fac_lapy),
#                        ((i,   j+1), + 1.0 * self.fac_lapy),
#                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
        if P != None:
            P.assemble()
        
#        if PETSc.COMM_WORLD.getRank() == 0:
#            print("     Matrix")
        
                
    

#    @cython.boundscheck(False)
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
        
