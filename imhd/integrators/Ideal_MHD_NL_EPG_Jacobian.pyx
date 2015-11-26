'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport Mat, Vec

from imhd.integrators.MHD_Derivatives import  MHD_Derivatives
from imhd.integrators.MHD_Derivatives cimport MHD_Derivatives



cdef class PETScJacobian(object):
    '''
    Scheme obtained by discrete Euler-Poincaré reduction by Gawlik et al.
    '''
    
    def __init__(self, object da1, object da5,
                 int nx, int ny,
                 double ht, double hx, double hy):
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
        self.fac_dt = 1.0 / 4. / self.ht
        
        self.fac_divx  = 1.0 / self.hx
        self.fac_divy  = 1.0 / self.hy
#        self.fac_divx  = 0.5 / self.hx
#        self.fac_divy  = 0.5 / self.hy
        
        self.fac_grdx  = 1.0 / 4. / self.hx
        self.fac_grdy  = 1.0 / 4. / self.hy
#        self.fac_grdx  = 0.5 / 4. / self.hx
#        self.fac_grdy  = 0.5 / 4. / self.hy
        
        
        # create history vectors
        self.Xh = self.da5.createGlobalVec()
        self.Xp = self.da5.createGlobalVec()
        
        # create local vectors
        self.localXh = da5.createLocalVec()
        self.localXp = da5.createLocalVec()
        
        # create derivatives object
        self.derivatives = MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A, Mat P = None):
        cdef np.int64_t i, j, ia, ja, ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da5.globalToLocal(self.Xp, self.localXp)
        self.da5.globalToLocal(self.Xh, self.localXh)
        
        xp = self.da5.getVecArray(self.localXp)[...]
        xh = self.da5.getVecArray(self.localXh)[...]
        
        cdef np.ndarray[double, ndim=2] Vxp = xp[:,:,0]
        cdef np.ndarray[double, ndim=2] Vyp = xp[:,:,1]
        cdef np.ndarray[double, ndim=2] Bxp = xp[:,:,2]
        cdef np.ndarray[double, ndim=2] Byp = xp[:,:,3]
        
        cdef np.ndarray[double, ndim=2] Vxh = xh[:,:,0]
        cdef np.ndarray[double, ndim=2] Vyh = xh[:,:,1]
        cdef np.ndarray[double, ndim=2] Bxh = xh[:,:,2]
        cdef np.ndarray[double, ndim=2] Byh = xh[:,:,3]
        
        cdef double[:,:] Vx_ave = 0.5 * (Vxp + Vxh)
        cdef double[:,:] Vy_ave = 0.5 * (Vyp + Vyh)
        cdef double[:,:] Bx_ave = 0.5 * (Bxp + Bxh)
        cdef double[:,:] By_ave = 0.5 * (Byp + Byh)

        cdef double[:,:] A_arr
        
        
        A.zeroEntries()
        
        if P != None:
            P.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            
            for j in range(ys, ye):
                jx = j-ys+2
                
                # V_x
                row.index = (i,j)
                row.field = 0
                
                # dt(dVx)
                # + psix_ux(Vx, Vy )
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
#                self.dt_x(A_arr, ix, jx)
                self.psix_ux(A_arr, Vxp, Vyp, ix, jx, +1)
#                self.psix_ux(A_arr, Vx_ave, Vy_ave, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # + psix_uy(Vx,  Vy)
                
                A_arr = np.zeros((5,5))
                
                self.psix_uy(A_arr, Vxp, Vyp, ix, jx, +1)
#                self.psix_uy(A_arr, Vx_ave, Vy_ave, ix, jx, -1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - psix_ux(Bx, By )
                
                A_arr = np.zeros((5,5))
                
                self.psix_ux(A_arr, Bxp, Byp, ix, jx, -1)
#                self.psix_ux(A_arr, Bx_ave, By_ave, ix, jx, +1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - psix_uy(Bx, By)
                
                A_arr = np.zeros((5,5))
                
                self.psix_uy(A_arr, Bxp, Byp, ix, jx, -1)
#                self.psix_uy(A_arr, Bx_ave, By_ave, ix, jx, +1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dx(P)
                
                col.field = 4
#                for index, value in [
#                        ((i+1, j-1), + 1. * self.fac_grdx),
#                        ((i,   j-1), - 1. * self.fac_grdx),
#                        ((i+1, j  ), + 2. * self.fac_grdx),
#                        ((i,   j  ), - 2. * self.fac_grdx),
#                        ((i+1, j+1), + 1. * self.fac_grdx),
#                        ((i,   j+1), - 1. * self.fac_grdx),
#                    ]:
#                for index, value in [
#                        ((i+1, j  ), + 4. * self.fac_grdx),
#                        ((i,   j  ), - 4. * self.fac_grdx),
#                    ]:
                for index, value in [
                        ((i,   j  ), + 4. * self.fac_grdx),
                        ((i-1, j  ), - 4. * self.fac_grdx),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # V_y
                row.index = (i,j)
                row.field = 1
                
                # + psiy_ux(Vx, Vy)
                
                A_arr = np.zeros((5,5))
                
                self.psiy_ux(A_arr, Vxp, Vyp, ix, jx, +1)
#                self.psiy_ux(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(Vy)
                # + psiy_uy(Vx, Vy)
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
#                self.dt_y(A_arr, ix, jx)
                self.psiy_uy(A_arr, Vxp, Vyp, ix, jx, +1)
#                self.psiy_uy(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # - psiy_ux(Bx, By)
                
                A_arr = np.zeros((5,5))
                
                self.psiy_ux(A_arr, Bxp, Byp, ix, jx, -1)
#                self.psiy_ux(A_arr, Bx_ave, By_ave, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - psiy_uy(Bx, By)
                
                A_arr = np.zeros((5,5))
                
                self.psiy_uy(A_arr, Bxp, Byp, ix, jx, -1)
#                self.psiy_uy(A_arr, Bx_ave, By_ave, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dy(P)
                col.field = 4
#                for index, value in [
#                        ((i-1, j+1), + 1. * self.fac_grdy),
#                        ((i-1, j  ), - 1. * self.fac_grdy),
#                        ((i,   j+1), + 2. * self.fac_grdy),
#                        ((i,   j  ), - 2. * self.fac_grdy),
#                        ((i+1, j+1), + 1. * self.fac_grdy),
#                        ((i+1, j  ), - 1. * self.fac_grdy),
#                    ]:
#                for index, value in [
#                        ((i,   j+1), + 4. * self.fac_grdy),
#                        ((i,   j  ), - 4. * self.fac_grdy),
#                    ]:
                for index, value in [
                        ((i,   j  ), + 4. * self.fac_grdy),
                        ((i,   j-1), - 4. * self.fac_grdy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # B_x
                row.index = (i,j)
                row.field = 2

#                # - phix(By,  dVx)
#                # - phix(Byh, dVx)
#
#                A_arr = np.zeros((5,5))
#                
#                self.phix_ux(A_arr, By_ave, ix, jx, -1)
#                
#                col.field = 0
#                for ia in range(0,5):
#                    for ja in range(0,5):
#                        col.index = (i-2+ia, j-2+ja)
#                        A.setValueStencil(row, col, A_arr[ia,ja])
#                
#                if P != None:
#                    P.setValueStencil(row, row, 1. / A_arr[2,2])
#                
#                
#                # + phix(Bx,  dVy)
#                # + phix(Bxh, dVy)
#
#                A_arr = np.zeros((5,5))
#                
#                self.phix_uy(A_arr, Bx_ave, ix, jx, +1)
#                
#                col.field = 1
#                for ia in range(0,5):
#                    for ja in range(0,5):
#                        col.index = (i-2+ia, j-2+ja)
#                        A.setValueStencil(row, col, A_arr[ia,ja])
#                
#                if P != None:
#                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # dt(dBx)
                # - phix(dBx, Vy )
                # - phix(dBx, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
#                self.dt_x(A_arr, ix, jx)
                self.phix_ux(A_arr, Vyh, ix, jx, -1)
#                self.phix_ux(A_arr, Vy_ave, ix, jx, +1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # + phix(Vx,  dBy)
                # + phix(Vxh, dBy)
                
                A_arr = np.zeros((5,5))
                
                self.phix_uy(A_arr, Vxh, ix, jx, +1)
#                self.phix_uy(A_arr, Vx_ave, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # B_y
                row.index = (i,j)
                row.field = 3
                
#                # + phiy(dVx, By )
#                # + phiy(dVx, Byh)
#                
#                A_arr = np.zeros((5,5))
#                
#                self.phiy_ux(A_arr, By_ave, ix, jx, +1)
#                
#                col.field = 0
#                for ia in range(0,5):
#                    for ja in range(0,5):
#                        col.index = (i-2+ia, j-2+ja)
#                        A.setValueStencil(row, col, A_arr[ia,ja])
#                
#                
#                # - phiy(Bx,  dVy)
#                # - phiy(Bxh, dVy)
#                
#                A_arr = np.zeros((5,5))
#                
#                self.phiy_uy(A_arr, Bx_ave, ix, jx, -1)
#                
#                col.field = 1
#                for ia in range(0,5):
#                    for ja in range(0,5):
#                        col.index = (i-2+ia, j-2+ja)
#                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - phiy(dBx, Vy )
                # - phiy(dBx, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.phiy_ux(A_arr, Vyh, ix, jx, -1)
#                self.phiy_ux(A_arr, Vy_ave, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(dBy)
                # + phiy(Vx,  dBy)
                # + phiy(Vxh, dBy)

                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
#                self.dt_y(A_arr, ix, jx)
                self.phiy_uy(A_arr, Vxh, ix, jx, +1)
#                self.phiy_uy(A_arr, Vx_ave, ix, jx, +1)
                
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
#                for index, value in [
#                        ((i,   j), + self.fac_divx),
#                        ((i-1, j), - self.fac_divx),
#                    ]:
                for index, value in [
                        ((i+1, j), + self.fac_divx),
                        ((i,   j), - self.fac_divx),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # dy(Vy)
                
                col.field = 1
#                for index, value in [
#                        ((i, j  ), + self.fac_divy),
#                        ((i, j-1), - self.fac_divy),
#                    ]:
                for index, value in [
                        ((i, j+1), + self.fac_divy),
                        ((i, j  ), - self.fac_divy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
        if P != None:
            P.assemble()
        
#        if PETSc.COMM_WORLD.getRank() == 0:
#            print("     Matrix")
        
                
    

    @cython.boundscheck(False)
    cdef double dt(self, double[:,:] A,
                                 int i, int j):
        
        # (i,   j  )
        A[2,2] += 4. * self.fac_dt
        
    
    
    @cython.boundscheck(False)
    cdef double dt_x(self, double[:,:] A,
                                 int i, int j):
        
        # (i,   j-1)
        A[2,1] += 1. * self.fac_dt
        
        # (i,   j  )
        A[2,2] += 2. * self.fac_dt
        
        # (i,   j+1)
        A[2,3] += 1. * self.fac_dt
        
    
    
    @cython.boundscheck(False)
    cdef double dt_y(self, double[:,:] A,
                                 int i, int j):
        
        # (i-1, j  )
        A[1,2] += 1. * self.fac_dt
        
        # (i,   j  )
        A[2,2] += 2. * self.fac_dt
        
        # (i+1, j  )
        A[3,2] += 1. * self.fac_dt
        
    
    



    @cython.boundscheck(False)
    cdef double rot(self, double[:,:] Ux,
                                double[:,:] Uy,
                                int i, int j):

        cdef double result
        
        result = ( \
                   + ( Uy[i, j] - Uy[i-1, j  ] ) / self.hx \
                   - ( Ux[i, j] - Ux[i,   j-1] ) / self.hy \
                 )
        
        return result



    cdef double psix_ux(self, double[:,:] A,
                                    double[:,:] Ux,
                                    double[:,:] Uy,
                                    int i, int j,
                                    double sign):
        
        cdef double fac = sign * 0.25 * 0.5
        
        # Ux[i,   j-1]
        A[2, 1] -= ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac / self.hy
        
        # Ux[i,   j  ]
        A[2, 2] += ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac / self.hy
        A[2, 2] -= ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac / self.hy
        
        # Ux[i,   j+1]
        A[2, 3] += ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac / self.hy
        
        

    cdef double psix_uy(self, double[:,:] A,
                                    double[:,:] Ux,
                                    double[:,:] Uy,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5
        
        # Uy[i-1, j  ]
        A[1, 2] += ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac / self.hx
        A[1, 2] -= self.rot(Ux, Uy, i,   j  ) * fac

        # Uy[i-1, j+1]
        A[1, 3] += ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac / self.hx
        A[1, 3] -= self.rot(Ux, Uy, i,   j+1) * fac 
        
 
        # Uy[i,   j  ]
        A[2, 2] -= ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac / self.hx
        A[2, 2] -= self.rot(Ux, Uy, i,   j  ) * fac
        
        # Uy[i,   j+1]
        A[2, 3] -= ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac / self.hx
        A[2, 3] -= self.rot(Ux, Uy, i,   j+1) * fac 
        

    cdef double psiy_ux(self, double[:,:] A,
                                    double[:,:] Ux,
                                    double[:,:] Uy,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5

        # Ux[i,   j-1]
        A[2, 1] += ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac / self.hy
        A[2, 1] += self.rot(Ux, Uy, i,   j  ) * fac

        # Ux[i,   j  ]
        A[2, 2] -= ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac / self.hy
        A[2, 2] += self.rot(Ux, Uy, i,   j  ) * fac 
        
        
        # Ux[i+1, j-1]
        A[3, 1] += ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac / self.hy
        A[3, 1] += self.rot(Ux, Uy, i+1, j  ) * fac 

        # Ux[i+1, j  ]
        A[3, 2] -= ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac / self.hy
        A[3, 2] += self.rot(Ux, Uy, i+1, j  ) * fac 


    cdef double psiy_uy(self, double[:,:] A,
                                    double[:,:] Ux,
                                    double[:,:] Uy,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5
        
        # Uy[i-1, j  ]
        A[1, 2] -= ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac / self.hx

        # Ux[i,   j  ]
        A[2, 2] += ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac / self.hx
        A[2, 2] -= ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac / self.hx
        
        
        # Ux[i+1, j  ]
        A[3, 2] += ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac / self.hx




    cdef double phix_ux(self, double[:,:] A,
                                    double[:,:] Fy,
                                    int i, int j,
                                    double sign):
        
        cdef double fac = sign * 0.25 * 0.5 / self.hy
        
        # Ux[i,   j-1]
        A[2, 1] += ( Fy[i-1, j  ] + Fy[i, j  ] ) * fac
        
        # Ux[i,   j  ]
        A[2, 2] -= ( Fy[i-1, j+1] + Fy[i, j+1] ) * fac
        A[2, 2] += ( Fy[i-1, j  ] + Fy[i, j  ] ) * fac
        
        # Ux[i,   j+1]
        A[2, 3] -= ( Fy[i-1, j+1] + Fy[i,   j+1] ) * fac
        
        

    cdef double phix_uy(self, double[:,:] A,
                                    double[:,:] Fx,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5 / self.hy
        
        # Uy[i-1, j  ]
        A[1, 2] += ( Fx[i,   j-1] + Fx[i,   j  ] ) * fac
        
        # Uy[i-1, j+1]
        A[1, 3] -= ( Fx[i,   j  ] + Fx[i,   j+1] ) * fac
        
        
        # Uy[i,   j  ]
        A[2, 2] += ( Fx[i,   j-1] + Fx[i,   j  ] ) * fac
        
        # Uy[i,   j+1]
        A[2, 3] -= ( Fx[i,   j  ] + Fx[i,   j+1] ) * fac
        


    cdef double phiy_ux(self, double[:,:] A,
                                    double[:,:] Fy,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5 / self.hx
        
        # Ux[i,   j-1]
        A[2, 1] -= ( Fy[i-1, j  ] + Fy[i,   j  ] ) * fac

        # Ux[i,   j  ]
        A[2, 2] -= ( Fy[i-1, j  ] + Fy[i,   j  ] ) * fac
        
        
        # Ux[i+1, j-1]
        A[3, 1] += ( Fy[i,   j  ] + Fy[i+1, j  ] ) * fac
        
        # Ux[i+1, j  ]
        A[3, 2] += ( Fy[i,   j  ] + Fy[i+1, j  ] ) * fac
        

    cdef double phiy_uy(self, double[:,:] A,
                                    double[:,:] Fx,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5 / self.hx
        
        # Uy[i-1, j  ]
        A[1, 2] -= ( Fx[i,   j-1] + Fx[i,   j  ] ) * fac
        
        # Uy[i,   j  ]
        A[2, 2] += ( Fx[i+1, j-1] + Fx[i+1, j  ] ) * fac
        A[2, 2] -= ( Fx[i,   j-1] + Fx[i,   j  ] ) * fac
        
        # Uy[i+1, j  ]
        A[3, 2] += ( Fx[i+1, j-1] + Fx[i+1, j  ] ) * fac


