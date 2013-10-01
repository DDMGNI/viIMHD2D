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
        
        self.ht_inv = 1. / ht
        self.hx_inv = 1. / hx
        self.hy_inv = 1. / hy
        
        # factors of derivatives
        self.fac_dt = 1. / 4. / self.ht
#        self.fac_dt = 1. / 2. / self.ht
        
        self.fac_divx  = 1.0 / self.hx
        self.fac_divy  = 1.0 / self.hy
        
#        self.fac_divx  = 0.5 / self.hx
#        self.fac_divy  = 0.5 / self.hy
        
#        self.fac_gradx = 1.0 / 4. / self.hx
#        self.fac_grady = 1.0 / 4. / self.hy
        
        self.fac_gradx = 0.5 / 4. / self.hx
        self.fac_grady = 0.5 / 4. / self.hy
        
        
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
                row.index = (i,j)
                row.field = 0
                
                # dt_x(dVx)
                # + 0.25 * dy(dVx, i, j  ) * ave_xt(Vyp, Vyh, i, j  )
                # + 0.25 * dy(dVx, i, j-1) * ave_xt(Vyp, Vyh, i, j-1)
                
                A_arr = np.zeros((3,3))
                
                self.dt_x(A_arr)
                self.dudy(A_arr,  0,  0, + 0.25 * self.ave_xt(Vyp, Vyh, ix, jx  ))
                self.dudy(A_arr,  0, -1, + 0.25 * self.ave_xt(Vyp, Vyh, ix, jx-1))
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                # + 0.25 * ave_xt(dVy, i, j  ) * dy(Vxp, i, j  )
                # + 0.25 * ave_xt(dVy, i, j  ) * dy(Vxh, i, j  )
                # + 0.25 * ave_xt(dVy, i, j-1) * dy(Vxp, i, j-1)
                # + 0.25 * ave_xt(dVy, i, j-1) * dy(Vxh, i, j-1)
                #
                # - 0.25 * ave_xt(dVy, i, j  ) * dx(Vyp, i, j  )
                # - 0.25 * ave_xt(dVy, i, j  ) * dx(Vyh, i, j  )
                # - 0.25 * ave_xt(dVy, i, j-1) * dx(Vyp, i, j-1)
                # - 0.25 * ave_xt(dVy, i, j-1) * dx(Vyh, i, j-1)
                #
                # - 0.25 * dx(dVy, i, j  ) * ave_xt(Vyp, Vyh, i, j  )
                # - 0.25 * dx(dVy, i, j-1) * ave_xt(Vyp, Vyh, i, j-1)
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_xt(A_arr,  0,  0, + 0.25 * (self.dy(Vxp, ix, jx  ) + self.dy(Vxh, ix, jx  )))
                self.u_ave_xt(A_arr,  0, -1, + 0.25 * (self.dy(Vxp, ix, jx-1) + self.dy(Vxh, ix, jx-1)))
                
                self.u_ave_xt(A_arr,  0,  0, - 0.25 * (self.dx(Vyp, ix, jx  ) + self.dx(Vyh, ix, jx  )))
                self.u_ave_xt(A_arr,  0, -1, - 0.25 * (self.dx(Vyp, ix, jx-1) + self.dx(Vyh, ix, jx-1)))
                
                self.dudx(A_arr,  0,  0, - 0.25 * self.ave_xt(Vyp, Vyh, ix, jx  ))
                self.dudx(A_arr,  0, -1, - 0.25 * self.ave_xt(Vyp, Vyh, ix, jx-1))
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - 0.25 * dy(dBx, i, j  ) * ave_xt(Byp, Byh, i, j  )
                # - 0.25 * dy(dBx, i, j-1) * ave_xt(Byp, Byh, i, j-1)
                
                A_arr = np.zeros((3,3))
                
                self.dudy(A_arr,  0,  0, - 0.25 * self.ave_xt(Byp, Byh, ix, jx  ))
                self.dudy(A_arr,  0, -1, - 0.25 * self.ave_xt(Byp, Byh, ix, jx-1))
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - 0.25 * ave_xt(dBy, i, j  ) * dy(Bxp, i, j  )
                # - 0.25 * ave_xt(dBy, i, j  ) * dy(Bxh, i, j  )
                # - 0.25 * ave_xt(dBy, i, j-1) * dy(Bxp, i, j-1)
                # - 0.25 * ave_xt(dBy, i, j-1) * dy(Bxh, i, j-1)
                #
                # + 0.25 * ave_xt(dBy, i, j  ) * dx(Byp, i, j  )
                # + 0.25 * ave_xt(dBy, i, j  ) * dx(Byh, i, j  )
                # + 0.25 * ave_xt(dBy, i, j-1) * dx(Byp, i, j-1)
                # + 0.25 * ave_xt(dBy, i, j-1) * dx(Byh, i, j-1)
                #
                # + 0.25 * dx(dBy, i, j  ) * ave_xt(Byp, Byh, i, j  )
                # + 0.25 * dx(dBy, i, j-1) * ave_xt(Byp, Byh, i, j-1)
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_xt(A_arr,  0,  0, - 0.25 * (self.dy(Bxp, ix, jx  ) + self.dy(Bxh, ix, jx  )))
                self.u_ave_xt(A_arr,  0, -1, - 0.25 * (self.dy(Bxp, ix, jx-1) + self.dy(Bxh, ix, jx-1)))
                
                self.u_ave_xt(A_arr,  0,  0, + 0.25 * (self.dx(Byp, ix, jx  ) + self.dx(Byh, ix, jx  )))
                self.u_ave_xt(A_arr,  0, -1, + 0.25 * (self.dx(Byp, ix, jx-1) + self.dx(Byh, ix, jx-1)))
                
                self.dudx(A_arr,  0,  0, + 0.25 * self.ave_xt(Byp, Byh, ix, jx  ))
                self.dudx(A_arr,  0, -1, + 0.25 * self.ave_xt(Byp, Byh, ix, jx-1))
                
                col.field = 3
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dx(P)
                
                col.field = 4
                for index, value in [
                        ((i+1, j-1), + 1. * self.fac_gradx),
                        ((i,   j-1), - 1. * self.fac_gradx),
                        ((i+1, j  ), + 2. * self.fac_gradx),
                        ((i,   j  ), - 2. * self.fac_gradx),
                        ((i+1, j+1), + 1. * self.fac_gradx),
                        ((i,   j+1), - 1. * self.fac_gradx),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # V_y
                row.index = (i,j)
                row.field = 1
                
                # - 0.25 * ave_yt(dVx, i,   j) * dy(Vxp, i,   j)
                # - 0.25 * ave_yt(dVx, i,   j) * dy(Vxh, i,   j)
                # - 0.25 * ave_yt(dVx, i-1, j) * dy(Vxp, i-1, j)
                # - 0.25 * ave_yt(dVx, i-1, j) * dy(Vxh, i-1, j)
                #
                # + 0.25 * ave_yt(dVx, i,   j) * dx(Vyp, i,   j)
                # + 0.25 * ave_yt(dVx, i,   j) * dx(Vyh, i,   j)
                # + 0.25 * ave_yt(dVx, i-1, j) * dx(Vyp, i-1, j)
                # + 0.25 * ave_yt(dVx, i-1, j) * dx(Vyh, i-1, j)
                #
                # - 0.25 * dy(dVx, i,   j) * ave_yt(Vxp, Vxh, i,   j)
                # - 0.25 * dy(dVx, i-1, j) * ave_yt(Vxp, Vxh, i-1, j)
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_yt(A_arr,  0,  0, - 0.25 * (self.dy(Vxp, ix,   jx) + self.dy(Vxh, ix,   jx)))
                self.u_ave_yt(A_arr, -1,  0, - 0.25 * (self.dy(Vxp, ix-1, jx) + self.dy(Vxh, ix-1, jx)))
                
                self.u_ave_yt(A_arr,  0,  0, + 0.25 * (self.dx(Vyp, ix,   jx) + self.dx(Vyh, ix,   jx)))
                self.u_ave_yt(A_arr, -1,  0, + 0.25 * (self.dx(Vyp, ix-1, jx) + self.dx(Vyh, ix-1, jx)))
                
                self.dudy(A_arr,  0,  0, - 0.25 * self.ave_yt(Vxp, Vxh, ix,   jx))
                self.dudy(A_arr, -1,  0, - 0.25 * self.ave_yt(Vxp, Vxh, ix-1, jx))
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt_y(dVy)
                #
                # + 0.25 * dx(dVy, i,   j) * ave_yt(Vxp, Vxh, i,   j)
                # + 0.25 * dx(dVy, i-1, j) * ave_yt(Vxp, Vxh, i-1, j)
                
                A_arr = np.zeros((3,3))
                
                self.dt_y(A_arr)
                self.dudx(A_arr,  0,  0, + 0.25 * self.ave_yt(Vxp, Vxh, ix,   jx))
                self.dudx(A_arr, -1,  0, + 0.25 * self.ave_yt(Vxp, Vxh, ix-1, jx))
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # + 0.25 * ave_yt(dBx, i,   j) * dy(Bxp, i,   j)
                # + 0.25 * ave_yt(dBx, i,   j) * dy(Bxh, i,   j)
                # + 0.25 * ave_yt(dBx, i-1, j) * dy(Bxp, i-1, j)
                # + 0.25 * ave_yt(dBx, i-1, j) * dy(Bxh, i-1, j)
                
                # - 0.25 * ave_yt(dBx, i,   j) * dx(Byp, i,   j)
                # - 0.25 * ave_yt(dBx, i,   j) * dx(Byh, i,   j)
                # - 0.25 * ave_yt(dBx, i-1, j) * dx(Byp, i-1, j)
                # - 0.25 * ave_yt(dBx, i-1, j) * dx(Byh, i-1, j)
                
                # + 0.25 * dy(dBx, i,   j) * ave_yt(Bxp, Bxh, i,   j)
                # + 0.25 * dy(dBx, i-1, j) * ave_yt(Bxp, Bxh, i-1, j)
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_yt(A_arr,  0,  0, + 0.25 * (self.dy(Bxp, ix,   jx) + self.dy(Bxh, ix,   jx)))
                self.u_ave_yt(A_arr, -1,  0, + 0.25 * (self.dy(Bxp, ix-1, jx) + self.dy(Bxh, ix-1, jx)))
                
                self.u_ave_yt(A_arr,  0,  0, - 0.25 * (self.dx(Byp, ix,   jx) + self.dx(Byh, ix,   jx)))
                self.u_ave_yt(A_arr, -1,  0, - 0.25 * (self.dx(Byp, ix-1, jx) + self.dx(Byh, ix-1, jx)))
                
                self.dudy(A_arr,  0,  0, + 0.25 * self.ave_yt(Bxp, Bxh, ix,   jx))
                self.dudy(A_arr, -1,  0, + 0.25 * self.ave_yt(Bxp, Bxh, ix-1, jx))
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - 0.25 * dx(dBy, i,   j) * ave_yt(Bxp, Bxh, i,   j)
                # - 0.25 * dx(dBy, i-1, j) * ave_yt(Bxp, Bxh, i-1, j)
                
                A_arr = np.zeros((3,3))
                
                self.dudx(A_arr,  0,  0, - 0.25 * self.ave_yt(Bxp, Bxh, ix,   jx))
                self.dudx(A_arr, -1,  0, - 0.25 * self.ave_yt(Bxp, Bxh, ix-1, jx))
                
                col.field = 3
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dy(P)
                col.field = 4
                for index, value in [
                        ((i-1, j+1), + 1. * self.fac_grady),
                        ((i-1, j  ), - 1. * self.fac_grady),
                        ((i,   j+1), + 2. * self.fac_grady),
                        ((i,   j  ), - 2. * self.fac_grady),
                        ((i+1, j+1), + 1. * self.fac_grady),
                        ((i+1, j  ), - 1. * self.fac_grady),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # B_x
                row.index = (i,j)
                row.field = 2
                
                # - ave_yt(dVx, i, j  ) * ave_xt(Byp, Byh, i, j  ) * self.hy_inv
                # + ave_yt(dVx, i, j-1) * ave_xt(Byp, Byh, i, j-1) * self.hy_inv
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_yt(A_arr,  0,  0, - self.ave_xt(Byp, Byh, ix, jx  ) * self.hy_inv)
                self.u_ave_yt(A_arr,  0, -1, + self.ave_xt(Byp, Byh, ix, jx-1) * self.hy_inv)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])


                # + ave_xt(dVy, i, j  ) * ave_yt(Bxp, Bxh, i, j  ) * self.hy_inv
                # - ave_xt(dVy, i, j-1) * ave_yt(Bxp, Bxh, i, j-1) * self.hy_inv
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_xt(A_arr,  0,  0, + self.ave_yt(Bxp, Bxh, ix, jx  ) * self.hy_inv)
                self.u_ave_xt(A_arr,  0, -1, - self.ave_yt(Bxp, Bxh, ix, jx-1) * self.hy_inv)
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt_x(dBx)
                #
                # + ave_yt(dBx, i, j  ) * ave_xt(Vyp, Vyh, i, j  ) * self.hy_inv
                # - ave_yt(dBx, i, j-1) * ave_xt(Vyp, Vyh, i, j-1) * self.hy_inv
                
                A_arr = np.zeros((3,3))
                
                self.dt_x(A_arr)
                self.u_ave_yt(A_arr,  0,  0, + self.ave_xt(Vyp, Vyh, ix, jx  ) * self.hy_inv)
                self.u_ave_yt(A_arr,  0, -1, - self.ave_xt(Vyp, Vyh, ix, jx-1) * self.hy_inv)
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[1,1])
                
                
                # - ave_xt(dBy, i, j  ) * ave_yt(Vxp, Vxh, i, j  ) * self.hy_inv
                # + ave_xt(dBy, i, j-1) * ave_yt(Vxp, Vxh, i, j-1) * self.hy_inv
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_xt(A_arr,  0,  0, - self.ave_yt(Vxp, Vxh, ix, jx  ) * self.hy_inv)
                self.u_ave_xt(A_arr,  0, -1, + self.ave_yt(Vxp, Vxh, ix, jx-1) * self.hy_inv)
                
                col.field = 3
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # B_y
                row.index = (i,j)
                row.field = 3
                
                # + ave_yt(dVx, i,   j) * ave_xt(Byp, Byh, i,   j) * self.hx_inv
                # - ave_yt(dVx, i-1, j) * ave_xt(Byp, Byh, i-1, j) * self.hx_inv
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_yt(A_arr,  0,  0, + self.ave_xt(Byp, Byh, ix,   jx) * self.hx_inv)
                self.u_ave_yt(A_arr, -1,  0, - self.ave_xt(Byp, Byh, ix-1, jx) * self.hx_inv)
                
                col.field = 0
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - ave_xt(dVy, i,   j) * ave_yt(Bxp, Bxh, i,   j) * self.hx_inv
                # + ave_xt(dVy, i-1, j) * ave_yt(Bxp, Bxh, i-1, j) * self.hx_inv
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_xt(A_arr,  0,  0, - self.ave_yt(Bxp, Bxh, ix,   jx) * self.hx_inv)
                self.u_ave_xt(A_arr, -1,  0, + self.ave_yt(Bxp, Bxh, ix-1, jx) * self.hx_inv)
                
                col.field = 1
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - ave_yt(dBx, i,   j) * ave_xt(Vyp, Vyh, i,   j) * self.hx_inv
                # + ave_yt(dBx, i-1, j) * ave_xt(Vyp, Vyh, i-1, j) * self.hx_inv
                
                A_arr = np.zeros((3,3))
                
                self.u_ave_yt(A_arr,  0,  0, - self.ave_xt(Vyp, Vyh, ix,   jx) * self.hx_inv)
                self.u_ave_yt(A_arr, -1,  0, + self.ave_xt(Vyp, Vyh, ix-1, jx) * self.hx_inv)
                
                col.field = 2
                for ia in range(0,3):
                    for ja in range(0,3):
                        col.index = (i-1+ia, j-1+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt_y(dBy)
                #
                # + ave_xt(dBy, i,   j) * ave_yt(Vxp, Vxh, i,   j) * self.hx_inv
                # - ave_xt(dBy, i-1, j) * ave_yt(Vxp, Vxh, i-1, j) * self.hx_inv

                A_arr = np.zeros((3,3))
                
                self.dt_y(A_arr)
                self.u_ave_xt(A_arr,  0,  0, + self.ave_yt(Vxp, Vxh, ix,   jx) * self.hx_inv)
                self.u_ave_xt(A_arr, -1,  0, - self.ave_yt(Vxp, Vxh, ix-1, jx) * self.hx_inv)
                
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
    cdef np.float64_t dt_x(self, np.ndarray[np.float64_t, ndim=2] A):
        
        # (i,   j-1)
        A[1,0] += 1. * self.fac_dt
        
        # (i,   j  )
        A[1,1] += 2. * self.fac_dt
        
        # (i,   j+1)
        A[1,2] += 1. * self.fac_dt
        
        
#        # (i,   j  )
#        A[1,1] += self.fac_dt
#        
#        # (i,   j+1)
#        A[1,2] += self.fac_dt
        

    @cython.boundscheck(False)
    cdef np.float64_t dt_y(self, np.ndarray[np.float64_t, ndim=2] A):
        
        # (i-1, j  )
        A[0,1] += 1. * self.fac_dt
        
        # (i,   j  )
        A[1,1] += 2. * self.fac_dt
        
        # (i+1, j  )
        A[2,1] += 1. * self.fac_dt

        
#        # (i,   j  )
#        A[1,1] += self.fac_dt
#        
#        # (i+1, j  )
#        A[2,1] += self.fac_dt
        
        

    @cython.boundscheck(False)
    cdef np.float64_t dudx(self, np.ndarray[np.float64_t, ndim=2] A,
                                 np.uint64_t i, np.uint64_t j,
                                 np.float64_t factor):
        
        A[i+1, j+1] += - self.hx_inv * factor       # (i,   j  )
        A[i+2, j+1] += + self.hx_inv * factor       # (i+1, j  )
        
        

    @cython.boundscheck(False)
    cdef np.float64_t dudy(self, np.ndarray[np.float64_t, ndim=2] A,
                                 np.uint64_t i, np.uint64_t j,
                                 np.float64_t factor):
        
        A[i+1, j+1] += - self.hy_inv * factor       # (i,   j  )         
        A[i+1, j+2] += + self.hy_inv * factor       # (i,   j+1)
        
        
        
        
    @cython.boundscheck(False)
    cdef np.float64_t u_ave_xt(self, np.ndarray[np.float64_t, ndim=2] A,
                                     np.uint64_t i, np.uint64_t j,
                                     np.float64_t factor):
        
        A[i+1, j+1] += 0.25 * factor                     # (i,   j)
        A[i+2, j+1] += 0.25 * factor                     # (i+1, j)
        
    
    @cython.boundscheck(False)
    cdef np.float64_t u_ave_yt(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t factor):
        
        A[i+1, j+1] += 0.25 * factor                     # (i, j  )
        A[i+1, j+2] += 0.25 * factor                     # (i, j+1)
        


    @cython.boundscheck(False)
    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_x
        '''
        
        return ( x[i+1, j] - x[i, j] ) * self.hx_inv
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_y
        '''
        
        return ( x[i, j+1] - x[i, j] ) * self.hy_inv
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t ave_xt(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] xh,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Average in x and t
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                   + x [i,   j] \
                   + x [i+1, j] \
                   + xh[i,   j] \
                   + xh[i+1, j] \
                 )
        
        return result


    @cython.boundscheck(False)
    cdef np.float64_t ave_yt(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] xh,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Average in y and t
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                   + x [i, j  ] \
                   + x [i, j+1] \
                   + xh[i, j  ] \
                   + xh[i, j+1] \
                 )
        
        return result


