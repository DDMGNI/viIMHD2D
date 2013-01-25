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
        self.fac_dt = 1.   / 64.  / self.ht
        self.fac_dx = 0.25 / 128. / self.hx
        self.fac_dy = 0.25 / 128. / self.hy
        
        self.fac_divx  = 1.0  / self.hx
        self.fac_divy  = 1.0  / self.hy
#        self.fac_divx  = 0.5  / self.hx
#        self.fac_divy  = 0.5  / self.hy
        
        self.fac_grdx    = 1.0 / 8. / self.hx
        self.fac_grdy    = 1.0 / 8. / self.hy
#        self.fac_grdx    = 0.5 / 8. / self.hx
#        self.fac_grdy    = 0.5 / 8. / self.hy
        
        
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
                # + fx_dx_ux(Vxp, dVx)
                # + fx_dx_ux(Vxh, dVx)
                # + fy_dy_ux(Vyp, dVx)
                # + fy_dy_ux(Vyh, dVx)
                # + ux_dx_fx(dVx, Vxp)
                # + ux_dx_fx(dVx, Vxh)
                
                A_arr = np.zeros((5,5))
                
                self.dt_x(A_arr, ix, jx)
                self.fx_dx_ux(A_arr, Vxp, ix, jx, +1)
                self.fx_dx_ux(A_arr, Vxh, ix, jx, +1)
                self.fy_dy_ux(A_arr, Vyp, ix, jx, +1)
                self.fy_dy_ux(A_arr, Vyh, ix, jx, +1)
                self.ux_dx_fx(A_arr, Vxp, ix, jx, +1)
                self.ux_dx_fx(A_arr, Vxh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # + uy_dy_fx(dVy, Vxp)
                # + uy_dy_fx(dVy, Vxh)
                
                A_arr = np.zeros((5,5))
                
                self.uy_dy_fx(A_arr, Vxp, ix, jx, +1)
                self.uy_dy_fx(A_arr, Vxh, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - fx_dx_ux(Bxp, dBx)
                # - fx_dx_ux(Bxh, dBx)
                # - fy_dy_ux(Byp, dBx)
                # - fy_dy_ux(Byh, dBx)
                # - ux_dx_fx(dBx, Bxp)
                # - ux_dx_fx(dBx, Bxh)
                
                A_arr = np.zeros((5,5))
                
                self.fx_dx_ux(A_arr, Bxp, ix, jx, -1)
                self.fx_dx_ux(A_arr, Bxh, ix, jx, -1)
                self.fy_dy_ux(A_arr, Byp, ix, jx, -1)
                self.fy_dy_ux(A_arr, Byh, ix, jx, -1)
                self.ux_dx_fx(A_arr, Bxp, ix, jx, -1)
                self.ux_dx_fx(A_arr, Bxh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - uy_dy_fx(dBy, Bxp)
                # - uy_dy_fx(dBy, Bxh)
                
                A_arr = np.zeros((5,5))
                
                self.uy_dy_fx(A_arr, Bxp, ix, jx, -1)
                self.uy_dy_fx(A_arr, Bxh, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dx(P)
                
                col.field = 4
                for index, value in [
                        ((i+1, j  ), + 3. * self.fac_grdx),
                        ((i,   j  ), - 3. * self.fac_grdx),
                        ((i+1, j-1), + 1. * self.fac_grdx),
                        ((i,   j-1), - 1. * self.fac_grdx),
                        ((i+1, j+1), + 1. * self.fac_grdx),
                        ((i,   j+1), - 1. * self.fac_grdx),
                        ((i+2, j  ), + 1. * self.fac_grdx),
                        ((i-1, j  ), - 1. * self.fac_grdx),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # V_y
                row.index = (i,j)
                row.field = 1
                
                # + ux_dx_fy(dVx, Vyp)
                # + ux_dx_fy(dVx, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.ux_dx_fy(A_arr, Vyp, ix, jx, +1)
                self.ux_dx_fy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(Vy)
                # + fx_dx_uy(Vxp, Vy)
                # + fx_dx_uy(Vxh, Vy)
                # + fy_dy_uy(Vyp, Vy)
                # + fy_dy_uy(Vyh, Vy)
                # + uy_dy_fy(dVy, Vyp)
                # + uy_dy_fy(dVy, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.dt_y(A_arr, ix, jx)
                self.fx_dx_uy(A_arr, Vxp, ix, jx, +1)
                self.fx_dx_uy(A_arr, Vxh, ix, jx, +1)
                self.fy_dy_uy(A_arr, Vyp, ix, jx, +1)
                self.fy_dy_uy(A_arr, Vyh, ix, jx, +1)
                self.uy_dy_fy(A_arr, Vyp, ix, jx, +1)
                self.uy_dy_fy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # - ux_dx_fy(dBx, Byp)
                # - ux_dx_fy(dBx, Byh)
                
                A_arr = np.zeros((5,5))
                
                self.ux_dx_fy(A_arr, Byp, ix, jx, -1)
                self.ux_dx_fy(A_arr, Byh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - fx_dx_uy(Bxp, By)
                # - fx_dx_uy(Bxh, By)
                # - fy_dy_uy(Byp, By)
                # - fy_dy_uy(Byh, By)
                # - uy_dy_fy(dBy, Byp)
                # - uy_dy_fy(dBy, Byh)
                
                A_arr = np.zeros((5,5))
                
                self.fx_dx_uy(A_arr, Bxp, ix, jx, -1)
                self.fx_dx_uy(A_arr, Bxh, ix, jx, -1)
                self.fy_dy_uy(A_arr, Byp, ix, jx, -1)
                self.fy_dy_uy(A_arr, Byh, ix, jx, -1)
                self.uy_dy_fy(A_arr, Byp, ix, jx, -1)
                self.uy_dy_fy(A_arr, Byh, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dy(P)
                col.field = 4
                for index, value in [
                        ((i,   j+1), + 3. * self.fac_grdy),
                        ((i,   j  ), - 3. * self.fac_grdy),
                        ((i-1, j+1), + 1. * self.fac_grdy),
                        ((i-1, j  ), - 1. * self.fac_grdy),
                        ((i+1, j+1), + 1. * self.fac_grdy),
                        ((i+1, j  ), - 1. * self.fac_grdy),
                        ((i,   j+2), + 1. * self.fac_grdy),
                        ((i,   j-1), - 1. * self.fac_grdy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # B_x
                row.index = (i,j)
                row.field = 2
                
                # - dy_ux_fy(dVx, Byp)
                # - dy_ux_fy(dVx, Byh)
                
                A_arr = np.zeros((5,5))
                
                self.dy_ux_fy(A_arr, Byp, ix, jx, -1)
                self.dy_ux_fy(A_arr, Byh, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])


                # + dy_fx_uy(Bxp, dVy)
                # + dy_fx_uy(Bxh, dVy)
                
                A_arr = np.zeros((5,5))
                
                self.dy_fx_uy(A_arr, Bxp, ix, jx, +1)
                self.dy_fx_uy(A_arr, Bxh, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(dBx)
                # + dy_ux_fy(dBx, Vyp)
                # + dy_ux_fy(dBx, Vyh)
                
                A_arr = np.zeros((5,5))
                
#                self.dt_x(A_arr, ix, jx)
                self.dt(A_arr, ix, jx)
                self.dy_ux_fy(A_arr, Vyp, ix, jx, +1)
                self.dy_ux_fy(A_arr, Vyh, ix, jx, +1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # - dy_fx_uy(Vxp, dBy)
                # - dy_fx_uy(Vxh, dBy)
                
                A_arr = np.zeros((5,5))
                
                self.dy_fx_uy(A_arr, Vxp, ix, jx, -1)
                self.dy_fx_uy(A_arr, Vxh, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # B_y
                row.index = (i,j)
                row.field = 3
                
                # + dx_ux_fy(dVx, Byp)
                # + dx_ux_fy(dVx, Byh)
                
                A_arr = np.zeros((5,5))
                
                self.dx_ux_fy(A_arr, Byp, ix, jx, +1)
                self.dx_ux_fy(A_arr, Byh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - dx_fx_uy(Bxp, dVy)
                # - dx_fx_uy(Bxh, dVy)
                
                A_arr = np.zeros((5,5))
                
                self.dx_fx_uy(A_arr, Bxp, ix, jx, -1)
                self.dx_fx_uy(A_arr, Bxh, ix, jx, -1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - dx_ux_fy(dBx, Vyp)
                # - dx_ux_fy(dBx, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.dx_ux_fy(A_arr, Vyp, ix, jx, -1)
                self.dx_ux_fy(A_arr, Vyh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(dBy)
                # + dx_fx_uy(Vxp, dBy)
                # + dx_fx_uy(Vxh, dBy)

                A_arr = np.zeros((5,5))
                
#                self.dt_y(A_arr, ix, jx)
                self.dt(A_arr, ix, jx)
                self.dx_fx_uy(A_arr, Vxp, ix, jx, +1)
                self.dx_fx_uy(A_arr, Vxh, ix, jx, +1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])

                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # P
                row.index = (i,j)
                row.field = 4
                
                # dx(Vx)
                
                col.field = 0
                for index, value in [
                        ((i,   j), + self.fac_divx),
                        ((i-1, j), - self.fac_divx),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # dy(Vy)
                
                col.field = 1
                for index, value in [
                        ((i, j  ), + self.fac_divy),
                        ((i, j-1), - self.fac_divy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
        if P != None:
            P.assemble()
        
#        if PETSc.COMM_WORLD.getRank() == 0:
#            print("     Matrix")
        
                
    

    @cython.boundscheck(False)
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] A,
                               np.uint64_t i, np.uint64_t j):
        
        # (i-1, j  )
        A[1,2] += 1. / 8. / self.ht
        
        # (i,   j-1)
        A[2,1] += 1. / 8. / self.ht
        
        # (i,   j  )
        A[2,2] += 4. / 8. / self.ht
        
        # (i,   j+1)
        A[2,3] += 1. / 8. / self.ht
        
        # (i+1, j  )
        A[3,2] += 1. / 8. / self.ht
        
    
    
    @cython.boundscheck(False)
    cdef np.float64_t dt_x(self, np.ndarray[np.float64_t, ndim=2] A,
                                 np.uint64_t i, np.uint64_t j):
        
        # (i-2, j  )
        A[0,2] += 1. * self.fac_dt
        
        
        # (i-1, j-1)
        A[1,1] += 2. * self.fac_dt
        
        # (i-1, j  )
        A[1,2] += 8. * self.fac_dt
        
        # (i-1, j+1)
        A[1,3] += 2. * self.fac_dt
        
        
        # (i,   j-2)
        A[2,0] += 1. * self.fac_dt
        
        # (i,   j-1)
        A[2,1] += 8. * self.fac_dt
        
        # (i,   j  )
        A[2,2] += 20. * self.fac_dt
        
        # (i,   j+1)
        A[2,3] += 8. * self.fac_dt
        
        # (i,   j+2)
        A[2,4] += 1. * self.fac_dt
        
        
        # (i+1, j-1)
        A[3,1] += 2. * self.fac_dt
        
        # (i+1, j  )
        A[3,2] += 8. * self.fac_dt
        
        # (i+1, j+1)
        A[3,3] += 2. * self.fac_dt
        
        
        # (i+2, j  )
        A[4,2] += 1. * self.fac_dt
        
    
    
    @cython.boundscheck(False)
    cdef np.float64_t dt_y(self, np.ndarray[np.float64_t, ndim=2] A,
                                 np.uint64_t i, np.uint64_t j):
        
        # (i-2, j  )
        A[0,2] += 1. * self.fac_dt
        
        
        # (i-1, j-1)
        A[1,1] += 2. * self.fac_dt
        
        # (i-1, j  )
        A[1,2] += 8. * self.fac_dt
        
        # (i-1, j+1)
        A[1,3] += 2. * self.fac_dt
        
        
        # (i,   j-2)
        A[2,0] += 1. * self.fac_dt
        
        # (i,   j-1)
        A[2,1] += 8. * self.fac_dt
        
        # (i,   j  )
        A[2,2] += 20. * self.fac_dt
        
        # (i,   j+1)
        A[2,3] += 8. * self.fac_dt
        
        # (i,   j+2)
        A[2,4] += 1. * self.fac_dt
        
        
        # (i+1, j-1)
        A[3,1] += 2. * self.fac_dt
        
        # (i+1, j  )
        A[3,2] += 8. * self.fac_dt
        
        # (i+1, j+1)
        A[3,3] += 2. * self.fac_dt
        
        
        # (i+2, j  )
        A[4,2] += 1. * self.fac_dt
        
    
    
    @cython.boundscheck(False)
    cdef np.float64_t fx_dx_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):
        
           
        # F[i-2, j  ]
        A[0, 2] += ( \
                    - 1. * F[i-2, j  ] \
                    - 1. * F[i-1, j-1] \
                    - 1. * F[i-1, j+1] \
                    - 4. * F[i-1, j  ] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j-1]
        A[1, 1] += ( \
                    - 1. * F[i-1, j-1] \
                    - 1. * F[i+1, j-1] \
                    - 1. * F[i,   j-2] \
                    - 4. * F[i,   j-1] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j  ]
        A[1, 2] += ( \
                    - 4. * F[i-1, j  ] \
                    - 4. * F[i+1, j  ] \
                    - 4. * F[i,   j-1] \
                    - 4. * F[i,   j+1] \
                    - 16 * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j+1]
        A[1, 3] += ( \
                    - 1. * F[i-1, j+1] \
                    - 1. * F[i+1, j+1] \
                    - 4. * F[i,   j+1] \
                    - 1. * F[i,   j+2] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i,   j  ]
        A[2, 2] += ( \
                    + 1. * F[i-2, j  ] \
                    + 1. * F[i-1, j-1] \
                    + 1. * F[i-1, j+1] \
                    + 4. * F[i-1, j  ] \
                    - 1. * F[i+1, j-1] \
                    - 1. * F[i+1, j+1] \
                    - 4. * F[i+1, j  ] \
                    - 1. * F[i+2, j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j-1]
        A[3, 1] += ( \
                    + 1. * F[i-1, j-1] \
                    + 1. * F[i+1, j-1] \
                    + 1. * F[i,   j-2] \
                    + 4. * F[i,   j-1] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j  ]
        A[3, 2] += ( \
                    + 4. * F[i-1, j  ] \
                    + 4. * F[i+1, j  ] \
                    + 4. * F[i,   j-1] \
                    + 4. * F[i,   j+1] \
                    + 16 * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j+1]
        A[3, 3] += ( \
                    + 1. * F[i-1, j+1] \
                    + 1. * F[i+1, j+1] \
                    + 4. * F[i,   j+1] \
                    + 1. * F[i,   j+2] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+2, j  ]
        A[4, 2] += ( \
                    + 1. * F[i+1, j-1] \
                    + 1. * F[i+1, j+1] \
                    + 4. * F[i+1, j  ] \
                    + 1. * F[i+2, j  ] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign

        


    @cython.boundscheck(False)
    cdef np.float64_t fx_dx_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):
        
           
           
        # F[i-2, j  ]
        A[0, 2] += ( \
                    - 2. * F[i-2, j+1] \
                    - 2. * F[i-2, j  ] \
                    - 2. * F[i-1, j+1] \
                    - 2. * F[i-1, j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j-1]
        A[1, 1] += ( \
                    - 2. * F[i-1, j-1] \
                    - 2. * F[i-1, j  ] \
                    - 2. * F[i,   j-1] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j  ]
        A[1, 2] += ( \
                    - 8. * F[i-1, j+1] \
                    - 8. * F[i-1, j  ] \
                    - 8. * F[i,   j+1] \
                    - 8. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j+1]
        A[1, 3] += ( \
                    - 2. * F[i-1, j+1] \
                    - 2. * F[i-1, j+2] \
                    - 2. * F[i,   j+1] \
                    - 2. * F[i,   j+2] \
                  ) * self.fac_dx * sign
        
        # F[i,   j  ]
        A[2, 2] += ( \
                    + 2. * F[i-2, j+1] \
                    + 2. * F[i-2, j  ] \
                    + 2. * F[i-1, j+1] \
                    + 2. * F[i-1, j  ] \
                    - 2. * F[i+1, j+1] \
                    - 2. * F[i+1, j  ] \
                    - 2. * F[i,   j+1] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j-1]
        A[3, 1] += ( \
                    + 2. * F[i-1, j-1] \
                    + 2. * F[i-1, j  ] \
                    + 2. * F[i,   j-1] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j  ]
        A[3, 2] += ( \
                    + 8. * F[i-1, j+1] \
                    + 8. * F[i-1, j  ] \
                    + 8. * F[i,   j+1] \
                    + 8. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j+1]
        A[3, 3] += ( \
                    + 2. * F[i-1, j+1] \
                    + 2. * F[i-1, j+2] \
                    + 2. * F[i,   j+1] \
                    + 2. * F[i,   j+2] \
                  ) * self.fac_dx * sign
        
        # F[i+2, j  ]
        A[4, 2] += ( \
                    + 2. * F[i+1, j+1] \
                    + 2. * F[i+1, j  ] \
                    + 2. * F[i,   j+1] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign


                  
        
    
    @cython.boundscheck(False)
    cdef np.float64_t fy_dy_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):
        
           
        # F[i-1, j-1]
        A[1, 1] += ( \
                    - 2. * F[i-1, j-1] \
                    - 2. * F[i-1, j  ] \
                    - 2. * F[i,   j-1] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i-1, j+1]
        A[1, 3] += ( \
                    + 2. * F[i-1, j-1] \
                    + 2. * F[i-1, j  ] \
                    + 2. * F[i,   j-1] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j-2]
        A[2, 0] += ( \
                    - 2. * F[i+1, j-2] \
                    - 2. * F[i+1, j-1] \
                    - 2. * F[i,   j-2] \
                    - 2. * F[i,   j-1] \
                  ) * self.fac_dy * sign
        
        # F[i,   j-1]
        A[2, 1] += ( \
                    - 8. * F[i+1, j-1] \
                    - 8. * F[i+1, j  ] \
                    - 8. * F[i,   j-1] \
                    - 8. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j  ]
        A[2, 2] += ( \
                    + 2. * F[i+1, j-2] \
                    + 2. * F[i+1, j-1] \
                    - 2. * F[i+1, j+1] \
                    - 2. * F[i+1, j  ] \
                    + 2. * F[i,   j-2] \
                    + 2. * F[i,   j-1] \
                    - 2. * F[i,   j+1] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j+1]
        A[2, 3] += ( \
                    + 8. * F[i+1, j-1] \
                    + 8. * F[i+1, j  ] \
                    + 8. * F[i,   j-1] \
                    + 8. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j+2]
        A[2, 4] += ( \
                    + 2. * F[i+1, j+1] \
                    + 2. * F[i+1, j  ] \
                    + 2. * F[i,   j+1] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j-1]
        A[3, 1] += ( \
                    - 2. * F[i+1, j-1] \
                    - 2. * F[i+1, j  ] \
                    - 2. * F[i+2, j-1] \
                    - 2. * F[i+2, j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j+1]
        A[3, 3] += ( \
                    + 2. * F[i+1, j-1] \
                    + 2. * F[i+1, j  ] \
                    + 2. * F[i+2, j-1] \
                    + 2. * F[i+2, j  ] \
                  ) * self.fac_dy * sign


                  
        
    
    @cython.boundscheck(False)
    cdef np.float64_t fy_dy_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):
        
        
        # F[i-1, j-1]
        A[1, 1] += ( \
                    - 1. * F[i-2, j  ] \
                    - 1. * F[i-1, j-1] \
                    - 1. * F[i-1, j+1] \
                    - 4. * F[i-1, j  ] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i-1, j+1]
        A[1, 3] += ( \
                    + 1. * F[i-2, j  ] \
                    + 1. * F[i-1, j-1] \
                    + 1. * F[i-1, j+1] \
                    + 4. * F[i-1, j  ] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j-2]
        A[2, 0] += ( \
                    - 1. * F[i-1, j-1] \
                    - 1. * F[i+1, j-1] \
                    - 1. * F[i,   j-2] \
                    - 4. * F[i,   j-1] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j-1]
        A[2, 1] += ( \
                    - 4. * F[i-1, j  ] \
                    - 4. * F[i+1, j  ] \
                    - 4. * F[i,   j-1] \
                    - 4. * F[i,   j+1] \
                    - 16 * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j  ]
        A[2, 2] += ( \
                    + 1. * F[i-1, j-1] \
                    - 1. * F[i-1, j+1] \
                    + 1. * F[i+1, j-1] \
                    - 1. * F[i+1, j+1] \
                    + 1. * F[i,   j-2] \
                    + 4. * F[i,   j-1] \
                    - 4. * F[i,   j+1] \
                    - 1. * F[i,   j+2] \
                  ) * self.fac_dy * sign
        
        # F[i,   j+1]
        A[2, 3] += ( \
                    + 4. * F[i-1, j  ] \
                    + 4. * F[i+1, j  ] \
                    + 4. * F[i,   j-1] \
                    + 4. * F[i,   j+1] \
                    + 16 * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j+2]
        A[2, 4] += ( \
                    + 1. * F[i-1, j+1] \
                    + 1. * F[i+1, j+1] \
                    + 4. * F[i,   j+1] \
                    + 1. * F[i,   j+2] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j-1]
        A[3, 1] += ( \
                    - 1. * F[i+1, j-1] \
                    - 1. * F[i+1, j+1] \
                    - 4. * F[i+1, j  ] \
                    - 1. * F[i+2, j  ] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j+1]
        A[3, 3] += ( \
                    + 1. * F[i+1, j-1] \
                    + 1. * F[i+1, j+1] \
                    + 4. * F[i+1, j  ] \
                    + 1. * F[i+2, j  ] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign

                  
        
    
    @cython.boundscheck(False)
    cdef np.float64_t ux_dx_fx(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):
        
           
        # F[i-2, j  ]
        A[0, 2] += ( \
                    - 1. * F[i-2, j  ] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j-1]
        A[1, 1] += ( \
                    - 1. * F[i-2, j  ] \
                    - 1. * F[i-1, j-1] \
                    + 1. * F[i+1, j-1] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j  ]
        A[1, 2] += ( \
                    - 4. * F[i-2, j  ] \
                    - 4. * F[i-1, j  ] \
                    + 4. * F[i+1, j  ] \
                    + 4. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j+1]
        A[1, 3] += ( \
                    - 1. * F[i-2, j  ] \
                    - 1. * F[i-1, j+1] \
                    + 1. * F[i+1, j+1] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i,   j-2]
        A[2, 0] += ( \
                    - 1. * F[i-1, j-1] \
                    + 1. * F[i+1, j-1] \
                  ) * self.fac_dx * sign
        
        # F[i,   j-1]
        A[2, 1] += ( \
                    - 4. * F[i-1, j-1] \
                    - 4. * F[i-1, j  ] \
                    + 4. * F[i+1, j-1] \
                    + 4. * F[i+1, j  ] \
                  ) * self.fac_dx * sign
        
        # F[i,   j  ]
        A[2, 2] += ( \
                    - 1. * F[i-2, j  ] \
                    - 1. * F[i-1, j-1] \
                    - 1. * F[i-1, j+1] \
                    - 16 * F[i-1, j  ] \
                    + 1. * F[i+1, j-1] \
                    + 1. * F[i+1, j+1] \
                    + 16 * F[i+1, j  ] \
                    + 1. * F[i+2, j  ] \
                  ) * self.fac_dx * sign
        
        # F[i,   j+1]
        A[2, 3] += ( \
                    - 4. * F[i-1, j+1] \
                    - 4. * F[i-1, j  ] \
                    + 4. * F[i+1, j+1] \
                    + 4. * F[i+1, j  ] \
                  ) * self.fac_dx * sign
        
        # F[i,   j+2]
        A[2, 4] += ( \
                    - 1. * F[i-1, j+1] \
                    + 1. * F[i+1, j+1] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j-1]
        A[3, 1] += ( \
                    - 1. * F[i-1, j-1] \
                    + 1. * F[i+1, j-1] \
                    + 1. * F[i+2, j  ] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j  ]
        A[3, 2] += ( \
                    - 4. * F[i-1, j  ] \
                    + 4. * F[i+1, j  ] \
                    + 4. * F[i+2, j  ] \
                    - 4. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j+1]
        A[3, 3] += ( \
                    - 1. * F[i-1, j+1] \
                    + 1. * F[i+1, j+1] \
                    + 1. * F[i+2, j  ] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+2, j  ]
        A[4, 2] += ( \
                    + 1. * F[i+2, j  ] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dx * sign

        
        
    
    @cython.boundscheck(False)
    cdef np.float64_t ux_dx_fy(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):
        
        # F[i-2, j  ]
        A[0, 2] += ( \
                    - 2. * F[i-2, j  ] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-2, j+1]
        A[0, 3] += ( \
                    - 2. * F[i-2, j  ] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j-1]
        A[1, 1] += ( \
                    - 2. * F[i-1, j-1] \
                    + 2. * F[i+1, j-1] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j  ]
        A[1, 2] += ( \
                    - 2. * F[i-2, j  ] \
                    - 2. * F[i-1, j-1] \
                    - 8. * F[i-1, j  ] \
                    + 2. * F[i+1, j-1] \
                    + 8. * F[i+1, j  ] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j+1]
        A[1, 3] += ( \
                    - 2. * F[i-2, j  ] \
                    - 2. * F[i-1, j+1] \
                    - 8. * F[i-1, j  ] \
                    + 2. * F[i+1, j+1] \
                    + 8. * F[i+1, j  ] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i-1, j+2]
        A[1, 4] += ( \
                    - 2. * F[i-1, j+1] \
                    + 2. * F[i+1, j+1] \
                  ) * self.fac_dx * sign
        
        # F[i,   j-1]
        A[2, 1] += ( \
                    - 2. * F[i-1, j-1] \
                    + 2. * F[i+1, j-1] \
                  ) * self.fac_dx * sign
        
        # F[i,   j  ]
        A[2, 2] += ( \
                    - 2. * F[i-1, j-1] \
                    - 8. * F[i-1, j  ] \
                    + 2. * F[i+1, j-1] \
                    + 8. * F[i+1, j  ] \
                    + 2. * F[i+2, j  ] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i,   j+1]
        A[2, 3] += ( \
                    - 2. * F[i-1, j+1] \
                    - 8. * F[i-1, j  ] \
                    + 2. * F[i+1, j+1] \
                    + 8. * F[i+1, j  ] \
                    + 2. * F[i+2, j  ] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i,   j+2]
        A[2, 4] += ( \
                    - 2. * F[i-1, j+1] \
                    + 2. * F[i+1, j+1] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j  ]
        A[3, 2] += ( \
                    + 2. * F[i+2, j  ] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        # F[i+1, j+1]
        A[3, 3] += ( \
                    + 2. * F[i+2, j  ] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dx * sign
        
        
    
    @cython.boundscheck(False)
    cdef np.float64_t uy_dy_fx(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):
        
           
        # F[i-1, j-1]
        A[1, 1] += ( \
                    - 2. * F[i-1, j-1] \
                    + 2. * F[i-1, j+1] \
                  ) * self.fac_dy * sign
        
        # F[i-1, j  ]
        A[1, 2] += ( \
                    - 2. * F[i-1, j-1] \
                    + 2. * F[i-1, j+1] \
                  ) * self.fac_dy * sign
        
        # F[i,   j-2]
        A[2, 0] += ( \
                    - 2. * F[i,   j-2] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j-1]
        A[2, 1] += ( \
                    - 2. * F[i-1, j-1] \
                    + 2. * F[i-1, j+1] \
                    - 2. * F[i,   j-2] \
                    - 8. * F[i,   j-1] \
                    + 8. * F[i,   j+1] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j  ]
        A[2, 2] += ( \
                    - 2. * F[i-1, j-1] \
                    + 2. * F[i-1, j+1] \
                    - 8. * F[i,   j-1] \
                    + 8. * F[i,   j+1] \
                    + 2. * F[i,   j+2] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j+1]
        A[2, 3] += ( \
                    + 2. * F[i,   j+2] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j-2]
        A[3, 0] += ( \
                    - 2. * F[i,   j-2] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j-1]
        A[3, 1] += ( \
                    - 2. * F[i+1, j-1] \
                    + 2. * F[i+1, j+1] \
                    - 2. * F[i,   j-2] \
                    - 8. * F[i,   j-1] \
                    + 8. * F[i,   j+1] \
                    + 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j  ]
        A[3, 2] += ( \
                    - 2. * F[i+1, j-1] \
                    + 2. * F[i+1, j+1] \
                    - 8. * F[i,   j-1] \
                    + 8. * F[i,   j+1] \
                    + 2. * F[i,   j+2] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j+1]
        A[3, 3] += ( \
                    + 2. * F[i,   j+2] \
                    - 2. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+2, j-1]
        A[4, 1] += ( \
                    - 2. * F[i+1, j-1] \
                    + 2. * F[i+1, j+1] \
                  ) * self.fac_dy * sign
        
        # F[i+2, j  ]
        A[4, 2] += ( \
                    - 2. * F[i+1, j-1] \
                    + 2. * F[i+1, j+1] \
                  ) * self.fac_dy * sign





    @cython.boundscheck(False)
    cdef np.float64_t uy_dy_fy(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):
        
        # F[i-2, j  ]
        A[0, 2] += ( \
                    - 1. * F[i-1, j-1] \
                    + 1. * F[i-1, j+1] \
                  ) * self.fac_dy * sign
        
        # F[i-1, j-1]
        A[1, 1] += ( \
                    - 1. * F[i-1, j-1] \
                    + 1. * F[i-1, j+1] \
                    - 1. * F[i,   j-2] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i-1, j  ]
        A[1, 2] += ( \
                    - 4. * F[i-1, j-1] \
                    + 4. * F[i-1, j+1] \
                    - 4. * F[i,   j-1] \
                    + 4. * F[i,   j+1] \
                  ) * self.fac_dy * sign
        
        # F[i-1, j+1]
        A[1, 3] += ( \
                    - 1. * F[i-1, j-1] \
                    + 1. * F[i-1, j+1] \
                    + 1. * F[i,   j+2] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j-2]
        A[2, 0] += ( \
                    - 1. * F[i,   j-2] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j-1]
        A[2, 1] += ( \
                    - 4. * F[i,   j-2] \
                    - 4. * F[i,   j-1] \
                    + 4. * F[i,   j+1] \
                    + 4. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j  ]
        A[2, 2] += ( \
                    - 1. * F[i-1, j-1] \
                    + 1. * F[i-1, j+1] \
                    - 1. * F[i+1, j-1] \
                    + 1. * F[i+1, j+1] \
                    - 1. * F[i,   j-2] \
                    - 16 * F[i,   j-1] \
                    + 16 * F[i,   j+1] \
                    + 1. * F[i,   j+2] \
                  ) * self.fac_dy * sign
        
        # F[i,   j+1]
        A[2, 3] += ( \
                    - 4. * F[i,   j-1] \
                    + 4. * F[i,   j+1] \
                    + 4. * F[i,   j+2] \
                    - 4. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i,   j+2]
        A[2, 4] += ( \
                    + 1. * F[i,   j+2] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j-1]
        A[3, 1] += ( \
                    - 1. * F[i+1, j-1] \
                    + 1. * F[i+1, j+1] \
                    - 1. * F[i,   j-2] \
                    + 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j  ]
        A[3, 2] += ( \
                    - 4. * F[i+1, j-1] \
                    + 4. * F[i+1, j+1] \
                    - 4. * F[i,   j-1] \
                    + 4. * F[i,   j+1] \
                  ) * self.fac_dy * sign
        
        # F[i+1, j+1]
        A[3, 3] += ( \
                    - 1. * F[i+1, j-1] \
                    + 1. * F[i+1, j+1] \
                    + 1. * F[i,   j+2] \
                    - 1. * F[i,   j  ] \
                  ) * self.fac_dy * sign
        
        # F[i+2, j  ]
        A[4, 2] += ( \
                    - 1. * F[i+1, j-1] \
                    + 1. * F[i+1, j+1] \
                  ) * self.fac_dy * sign





    cdef np.float64_t dx_fx_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):

        # U[i-2, j  ]
        A[0, 2] += ( \
                    - 1  * F[i-2, j+1] \
                    - 1  * F[i-2, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i-1, j-1]
        A[1, 1] += ( \
                    - 1  * F[i-1, j-1] \
                    - 1  * F[i-1, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i-1, j  ]
        A[1, 2] += ( \
                    - 1  * F[i-2, j+1] \
                    - 1  * F[i-2, j  ] \
                    - 3  * F[i-1, j+1] \
                    - 3  * F[i-1, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i-1, j+1]
        A[1, 3] += ( \
                    - 1  * F[i-1, j+1] \
                    - 1  * F[i-1, j+2] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i,   j-1]
        A[2, 1] += ( \
                    - 1  * F[i-1, j-1] \
                    - 1  * F[i-1, j  ] \
                    + 1  * F[i,   j-1] \
                    + 1  * F[i,   j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i,   j  ]
        A[2, 2] += ( \
                    - 3  * F[i-1, j+1] \
                    - 3  * F[i-1, j  ] \
                    + 3  * F[i,   j+1] \
                    + 3  * F[i,   j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i,   j+1]
        A[2, 3] += ( \
                    - 1  * F[i-1, j+1] \
                    - 1  * F[i-1, j+2] \
                    + 1  * F[i,   j+1] \
                    + 1  * F[i,   j+2] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i+1, j-1]
        A[3, 1] += ( \
                    + 1  * F[i,   j-1] \
                    + 1  * F[i,   j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i+1, j  ]
        A[3, 2] += ( \
                    + 1  * F[i+1, j+1] \
                    + 1  * F[i+1, j  ] \
                    + 3  * F[i,   j+1] \
                    + 3  * F[i,   j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i+1, j+1]
        A[3, 3] += ( \
                    + 1  * F[i,   j+1] \
                    + 1  * F[i,   j+2] \
                  ) * 4. * self.fac_dx * sign
        
        # U[i+2, j  ]
        A[4, 2] += ( \
                    + 1  * F[i+1, j+1] \
                    + 1  * F[i+1, j  ] \
                  ) * 4. * self.fac_dx * sign



    cdef np.float64_t dx_ux_fy(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):

        # F[i-2, j  ]
        A[0, 2] += (\
                    - 1 * F[i - 2, j  ] \
                    - 1 * F[i - 1, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i-2, j+1]
        A[0, 3] += (\
                    - 1 * F[i - 2, j  ] \
                    - 1 * F[i - 1, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i-1, j-1]
        A[1, 1] += (\
                    - 1 * F[i - 1, j - 1] \
                    - 1 * F[i, j - 1] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i-1, j  ]
        A[1, 2] += (\
                    - 1 * F[i - 1, j - 1] \
                    - 3 * F[i - 1, j  ] \
                    - 1 * F[i, j - 1] \
                    - 3 * F[i, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i-1, j+1]
        A[1, 3] += (\
                    - 1 * F[i - 1, j + 1] \
                    - 3 * F[i - 1, j  ] \
                    - 1 * F[i, j + 1] \
                    - 3 * F[i, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i-1, j+2]
        A[1, 4] += (\
                    - 1 * F[i - 1, j + 1] \
                    - 1 * F[i, j + 1] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i,   j-1]
        A[2, 1] += (\
                    + 1 * F[i + 1, j - 1] \
                    + 1 * F[i, j - 1] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i,   j  ]
        A[2, 2] += (\
                    + 1 * F[i + 1, j - 1] \
                    + 3 * F[i + 1, j  ] \
                    + 1 * F[i, j - 1] \
                    + 3 * F[i, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i,   j+1]
        A[2, 3] += (\
                    + 1 * F[i + 1, j + 1] \
                    + 3 * F[i + 1, j  ] \
                    + 1 * F[i, j + 1] \
                    + 3 * F[i, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i,   j+2]
        A[2, 4] += (\
                    + 1 * F[i + 1, j + 1] \
                    + 1 * F[i, j + 1] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i+1, j  ]
        A[3, 2] += (\
                    + 1 * F[i + 1, j  ] \
                    + 1 * F[i + 2, j  ] \
                  ) * 4. * self.fac_dx * sign
        
        # F[i+1, j+1]
        A[3, 3] += (\
                    + 1 * F[i + 1, j  ] \
                    + 1 * F[i + 2, j  ] \
                  ) * 4. * self.fac_dx * sign



    cdef np.float64_t dy_fx_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):

           
        # U[i-1, j-1]
        A[1, 1] += ( \
                    - 1  * F[i-1, j-1] \
                    - 1  * F[i-1, j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i-1, j  ]
        A[1, 2] += ( \
                    + 1  * F[i-1, j+1] \
                    + 1  * F[i-1, j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i,   j-2]
        A[2, 0] += ( \
                    - 1  * F[i,   j-2] \
                    - 1  * F[i,   j-1] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i,   j-1]
        A[2, 1] += ( \
                    - 1  * F[i-1, j-1] \
                    - 1  * F[i-1, j  ] \
                    - 3  * F[i,   j-1] \
                    - 3  * F[i,   j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i,   j  ]
        A[2, 2] += ( \
                    + 1  * F[i-1, j+1] \
                    + 1  * F[i-1, j  ] \
                    + 3  * F[i,   j+1] \
                    + 3  * F[i,   j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i,   j+1]
        A[2, 3] += ( \
                    + 1  * F[i,   j+1] \
                    + 1  * F[i,   j+2] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i+1, j-2]
        A[3, 0] += ( \
                    - 1  * F[i,   j-2] \
                    - 1  * F[i,   j-1] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i+1, j-1]
        A[3, 1] += ( \
                    - 1  * F[i+1, j-1] \
                    - 1  * F[i+1, j  ] \
                    - 3  * F[i,   j-1] \
                    - 3  * F[i,   j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i+1, j  ]
        A[3, 2] += ( \
                    + 1  * F[i+1, j+1] \
                    + 1  * F[i+1, j  ] \
                    + 3  * F[i,   j+1] \
                    + 3  * F[i,   j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i+1, j+1]
        A[3, 3] += ( \
                    + 1  * F[i,   j+1] \
                    + 1  * F[i,   j+2] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i+2, j-1]
        A[4, 1] += ( \
                    - 1  * F[i+1, j-1] \
                    - 1  * F[i+1, j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # U[i+2, j  ]
        A[4, 2] += ( \
                    + 1  * F[i+1, j+1] \
                    + 1  * F[i+1, j  ] \
                  ) * 4. * self.fac_dy * sign



    cdef np.float64_t dy_ux_fy(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.ndarray[np.float64_t, ndim=2] F,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t sign):

           
        # F[i-1, j-1]
        A[1, 1] += ( \
                    - 1  * F[i-1, j-1] \
                    - 1  * F[i,   j-1] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i-1, j  ]
        A[1, 2] += ( \
                    - 1  * F[i-1, j-1] \
                    + 1  * F[i-1, j  ] \
                    - 1  * F[i,   j-1] \
                    + 1  * F[i,   j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i-1, j+1]
        A[1, 3] += ( \
                    + 1  * F[i-1, j  ] \
                    + 1  * F[i,   j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i,   j-2]
        A[2, 0] += ( \
                    - 1  * F[i+1, j-2] \
                    - 1  * F[i,   j-2] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i,   j-1]
        A[2, 1] += ( \
                    - 1  * F[i+1, j-2] \
                    - 3  * F[i+1, j-1] \
                    - 1  * F[i,   j-2] \
                    - 3  * F[i,   j-1] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i,   j  ]
        A[2, 2] += ( \
                    - 3  * F[i+1, j-1] \
                    + 3  * F[i+1, j  ] \
                    - 3  * F[i,   j-1] \
                    + 3  * F[i,   j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i,   j+1]
        A[2, 3] += ( \
                    + 1  * F[i+1, j+1] \
                    + 3  * F[i+1, j  ] \
                    + 1  * F[i,   j+1] \
                    + 3  * F[i,   j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i,   j+2]
        A[2, 4] += ( \
                    + 1  * F[i+1, j+1] \
                    + 1  * F[i,   j+1] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i+1, j-1]
        A[3, 1] += ( \
                    - 1  * F[i+1, j-1] \
                    - 1  * F[i+2, j-1] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i+1, j  ]
        A[3, 2] += ( \
                    - 1  * F[i+1, j-1] \
                    + 1  * F[i+1, j  ] \
                    - 1  * F[i+2, j-1] \
                    + 1  * F[i+2, j  ] \
                  ) * 4. * self.fac_dy * sign
        
        # F[i+1, j+1]
        A[3, 3] += ( \
                    + 1  * F[i+1, j  ] \
                    + 1  * F[i+2, j  ] \
                  ) * 4. * self.fac_dy * sign



