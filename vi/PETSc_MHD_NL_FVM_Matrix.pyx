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
        self.fac_dt = 1.0 / self.ht
        
        self.fac_grdx  = 1.0 / self.hx
        self.fac_grdy  = 1.0 / self.hy
        
        self.fac_divx  = 1.0 / self.hx
        self.fac_divy  = 1.0 / self.hy
        
        
        # create history vectors
        self.Xh = self.da5.createGlobalVec()
        
        # create local vectors
        self.localXh = da5.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A, Mat P = None):
        cdef np.int64_t i, j, ia, ja, ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da5.globalToLocal(self.Xh, self.localXh)
        
        xh = self.da5.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,3]

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
                # + psix_ux(Vxh, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
                self.psix_ux(A_arr, Vxh, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # + psix_uy(Vxh, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.psix_uy(A_arr, Vxh, Vyh, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - psix_ux(Bxh, Byh)
                
                A_arr = np.zeros((5,5))
                
                self.psix_ux(A_arr, Bxh, Byh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - psix_uy(Bxh, Byh)
                
                A_arr = np.zeros((5,5))
                
                self.psix_uy(A_arr, Bxh, Byh, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dx(P)
                
                col.field = 4
                for index, value in [
                        ((i,   j  ), + self.fac_grdx),
                        ((i-1, j  ), - self.fac_grdx),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # V_y
                row.index = (i,j)
                row.field = 1
                
                # + psiy_ux(Vxh, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.psiy_ux(A_arr, Vxh, Vyh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(Vy)
                # + psiy_uy(Vxh, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
                self.psiy_uy(A_arr, Vxh, Vyh, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # - psiy_ux(Bxh, Byh)
                
                A_arr = np.zeros((5,5))
                
#                self.psiy_ux(A_arr, Bxh, Byh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - psiy_uy(Bxh, Byh)
                
                A_arr = np.zeros((5,5))
                
                self.psiy_uy(A_arr, Bxh, Byh, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dy(P)
                col.field = 4
                for index, value in [
                        ((i,   j  ), + self.fac_grdy),
                        ((i,   j-1), - self.fac_grdy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # B_x
                row.index = (i,j)
                row.field = 2

                # + phix(Byh, dVx)

                A_arr = np.zeros((5,5))
                
                self.phix_ux(A_arr, Byh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # - phix(Bxh, dVy)

                A_arr = np.zeros((5,5))
                
#                self.phix_uy(A_arr, Bxh, ix, jx, -1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # dt(dBx)
                # - phix(dBx, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
                self.phix_ux(A_arr, Vyh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # + phix(Vxh, dBy)
                
                A_arr = np.zeros((5,5))
                
                self.phix_uy(A_arr, Vxh, ix, jx, +1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # B_y
                row.index = (i,j)
                row.field = 3
                
                # + phiy(dVx, Byh)
                
                A_arr = np.zeros((5,5))
                
#                self.phiy_ux(A_arr, Byh, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - phiy(Bxh, dVy)
                
                A_arr = np.zeros((5,5))
                
#                self.phiy_uy(A_arr, Bxh, ix, jx, -1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - phiy(dBx, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.phiy_ux(A_arr, Vyh, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(dBy)
                # + phiy(Vxh, dBy)

                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
                self.phiy_uy(A_arr, Vxh, ix, jx, +1)
                
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
                        ((i+1, j), + self.fac_divx),
                        ((i,   j), - self.fac_divx),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # dy(Vy)
                
                col.field = 1
                for index, value in [
                        ((i, j+1), + self.fac_divy),
                        ((i, j  ), - self.fac_divy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
        if P != None:
            P.assemble()
        
                
    
    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=3] y = self.da5.getVecArray(B)[...]

        self.da5.globalToLocal(self.Xh, self.localXh)
        
        xh = self.da5.getVecArray(self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] Ph  = xh[...][:,:,4]

        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = self.dt(Vxh, ix, jx) #\
#                             - 0.5 * self.derivatives.psix(Vxh, Vyh, ix, jx) #\
#                             + self.derivatives.psix(Bxh, Byh, ix, jx)
                
                # V_y
                y[iy, jy, 1] = self.dt(Vyh, ix, jx) #\
#                             - 0.5 * self.derivatives.psix(Vxh, Vyh, ix, jx) #\
#                             + self.derivatives.psix(Bxh, Byh, ix, jx)
                              
                # B_x
                y[iy, jy, 2] = self.dt(Bxh, ix, jx) #\
#                             - 0.5 * self.derivatives.phix(Vxh, Byh, ix, jx) \
#                             + 0.5 * self.derivatives.phix(Bxh, Vyh, ix, jx)
                    
                # B_y
                y[iy, jy, 3] = self.dt(Byh, ix, jx) #\
#                             - 0.5 * self.derivatives.phiy(Vxh, Byh, ix, jx) \
#                             + 0.5 * self.derivatives.phiy(Bxh, Vyh, ix, jx)
                
                # P
                y[iy, jy, 4] = 0. #- 0.5 * self.derivatives.gradx_simple(Vxh, ix, jx) \
#                               - 0.5 * self.derivatives.grady_simple(Vyh, ix, jx)
                             




    @cython.boundscheck(False)
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] A,
                                 np.uint64_t i, np.uint64_t j):
        
        # (i,   j  )
        A[2,2] += self.fac_dt
        
    
    
    @cython.boundscheck(False)
    cdef np.float64_t rot(self, np.ndarray[np.float64_t, ndim=2] Ux,
                                np.ndarray[np.float64_t, ndim=2] Uy,
                                np.uint64_t i, np.uint64_t j):

        cdef np.float64_t result
        
        result = ( \
                   + ( Uy[i, j] - Uy[i-1, j  ] ) / self.hx \
                   - ( Ux[i, j] - Ux[i,   j-1] ) / self.hy \
                 )
        
        return result



    cdef np.float64_t psix_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Ux,
                                    np.ndarray[np.float64_t, ndim=2] Uy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign):
        
        cdef np.float64_t fac = sign * 0.25 * 0.5
        
#        # Ux[i,   j-1]
#        A[2, 1] -= ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac / self.hy
#        
#        # Ux[i,   j  ]
#        A[2, 2] += ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac / self.hy
#        A[2, 2] -= ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac / self.hy
#        
#        # Ux[i,   j+1]
#        A[2, 3] += ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac / self.hy
        
        

    cdef np.float64_t psix_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Ux,
                                    np.ndarray[np.float64_t, ndim=2] Uy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign):

        cdef np.float64_t fac = sign * 0.25
#        cdef np.float64_t fac = sign * 0.25 * 0.5
        
        # Uy[i-1, j  ]
#        A[1, 2] += ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac / self.hx
        A[1, 2] -= self.rot(Ux, Uy, i,   j  ) * fac

        # Uy[i-1, j+1]
#        A[1, 3] += ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac / self.hx
        A[1, 3] -= self.rot(Ux, Uy, i,   j+1) * fac
        
 
        # Uy[i,   j  ]
#        A[2, 2] -= ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac / self.hx
        A[2, 2] -= self.rot(Ux, Uy, i,   j  ) * fac
        
        # Uy[i,   j+1]
#        A[2, 3] -= ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac / self.hx
        A[2, 3] -= self.rot(Ux, Uy, i,   j+1) * fac
        

    cdef np.float64_t psiy_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Ux,
                                    np.ndarray[np.float64_t, ndim=2] Uy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign):

        cdef np.float64_t fac = sign * 0.25
#        cdef np.float64_t fac = sign * 0.25 * 0.5

        # Ux[i,   j-1]
#        A[2, 1] += ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac / self.hy
        A[2, 1] += self.rot(Ux, Uy, i,   j  ) * fac

        # Ux[i,   j  ]
#        A[2, 2] -= ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac / self.hy
        A[2, 2] += self.rot(Ux, Uy, i,   j  ) * fac
        
        
        # Ux[i+1, j-1]
#        A[3, 1] += ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac / self.hy
        A[3, 1] += self.rot(Ux, Uy, i+1, j  ) * fac

        # Ux[i+1, j  ]
#        A[3, 2] -= ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac / self.hy
        A[3, 2] += self.rot(Ux, Uy, i+1, j  ) * fac


    cdef np.float64_t psiy_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Ux,
                                    np.ndarray[np.float64_t, ndim=2] Uy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign):

        cdef np.float64_t fac = sign * 0.25 * 0.5
        
#        # Uy[i-1, j  ]
#        A[1, 2] -= ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac / self.hx
#
#        # Ux[i,   j  ]
#        A[2, 2] += ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac / self.hx
#        A[2, 2] -= ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac / self.hx
#        
#        
#        # Ux[i+1, j  ]
#        A[3, 2] += ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac / self.hx




    cdef np.float64_t phix_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Fy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign):
        
        cdef np.float64_t fac = sign * 0.25 / self.hy
#        cdef np.float64_t fac = sign * 0.25 * 0.5 / self.hy
        
        # Ux[i,   j-1]
        A[2, 1] += ( Fy[i-1, j  ] + Fy[i, j  ] ) * fac
        
        # Ux[i,   j  ]
        A[2, 2] -= ( Fy[i-1, j+1] + Fy[i, j+1] ) * fac
        A[2, 2] += ( Fy[i-1, j  ] + Fy[i, j  ] ) * fac
        
        # Ux[i,   j+1]
        A[2, 3] -= ( Fy[i-1, j+1] + Fy[i,   j+1] ) * fac
        
        

    cdef np.float64_t phix_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Fx,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign):

        cdef np.float64_t fac = sign * 0.25 / self.hy
#        cdef np.float64_t fac = sign * 0.25 * 0.5 / self.hy
        
        # Uy[i-1, j  ]
        A[1, 2] += ( Fx[i,   j-1] + Fx[i,   j  ] ) * fac
        
        # Uy[i-1, j+1]
        A[1, 3] -= ( Fx[i,   j  ] + Fx[i,   j+1] ) * fac
        
        
        # Uy[i,   j  ]
        A[2, 2] += ( Fx[i,   j-1] + Fx[i,   j  ] ) * fac
        
        # Uy[i,   j+1]
        A[2, 3] -= ( Fx[i,   j  ] + Fx[i,   j+1] ) * fac
        


    cdef np.float64_t phiy_ux(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Fy,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign):

        cdef np.float64_t fac = sign * 0.25 / self.hx
#        cdef np.float64_t fac = sign * 0.25 * 0.5 / self.hx
        
        # Ux[i,   j-1]
        A[2, 1] -= ( Fy[i-1, j  ] + Fy[i,   j  ] ) * fac

        # Ux[i,   j  ]
        A[2, 2] -= ( Fy[i-1, j  ] + Fy[i,   j  ] ) * fac
        
        
        # Ux[i+1, j-1]
        A[3, 1] += ( Fy[i,   j  ] + Fy[i+1, j  ] ) * fac
        
        # Ux[i+1, j  ]
        A[3, 2] += ( Fy[i,   j  ] + Fy[i+1, j  ] ) * fac
        

    cdef np.float64_t phiy_uy(self, np.ndarray[np.float64_t, ndim=2] A,
                                    np.ndarray[np.float64_t, ndim=2] Fx,
                                    np.uint64_t i, np.uint64_t j,
                                    np.float64_t sign):

        cdef np.float64_t fac = sign * 0.25 / self.hx
#        cdef np.float64_t fac = sign * 0.25 * 0.5 / self.hx
        
        # Uy[i-1, j  ]
        A[1, 2] -= ( Fx[i,   j-1] + Fx[i,   j  ] ) * fac
        
        # Uy[i,   j  ]
        A[2, 2] += ( Fx[i+1, j-1] + Fx[i+1, j  ] ) * fac
        A[2, 2] -= ( Fx[i,   j-1] + Fx[i,   j  ] ) * fac
        
        # Uy[i+1, j  ]
        A[3, 2] += ( Fx[i+1, j-1] + Fx[i+1, j  ] ) * fac



