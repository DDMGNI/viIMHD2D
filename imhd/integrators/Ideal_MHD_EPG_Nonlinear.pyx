'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport SNES, Mat, Vec

from imhd.integrators.MHD_Derivatives import  MHD_Derivatives
from imhd.integrators.MHD_Derivatives cimport MHD_Derivatives



cdef class PETScFunction(object):
    '''
    Scheme obtained by discrete Euler-Poincaré reduction by Gawlik et al.
    '''
    
    def __init__(self, object da1, object da7,
                 int nx, int ny,
                 double ht, double hx, double hy,
                 double mu, double nu, double eta):
        '''
        Constructor
        '''
        
        # distributed array
        self.da1 = da1
        self.da7 = da7
        
        # grid
        self.nx = nx
        self.ny = ny
        
        # grid size
        self.nx = nx
        self.ny = ny
        
        # step size
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        # factors of derivatives
        self.ht_inv = 1. / self.ht
        
        self.fac_divx  = 1.0 / self.hx
        self.fac_divy  = 1.0 / self.hy
        
        self.fac_grdx  = 1.0 / 4. / self.hx
        self.fac_grdy  = 1.0 / 4. / self.hy
        
        # friction, viscosity and resistivity (WARNING: NOT USED IN THIS SCHEME!)
        self.mu  = mu
        self.nu  = nu
        self.eta = eta
        
        # create history vector
        self.Xh = self.da7.createGlobalVec()
        self.Xp = self.da7.createGlobalVec()
        
        # create local vectors
        self.localX  = da7.createLocalVec()
        self.localXh = da7.createLocalVec()
        self.localXp = da7.createLocalVec()
        
        # create derivatives object
        self.derivatives = MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
            
    @cython.boundscheck(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da7.globalToLocal(X,       self.localX)
        self.da7.globalToLocal(self.Xh, self.localXh)
        
        x  = self.da7.getVecArray(self.localX) [...]
        xh = self.da7.getVecArray(self.localXh)[...]
        
        cdef np.ndarray[double, ndim=3] y = self.da7.getVecArray(Y)[...]

        cdef np.ndarray[double, ndim=2] Vx  = x [:,:,0]
        cdef np.ndarray[double, ndim=2] Vy  = x [:,:,1]
        cdef np.ndarray[double, ndim=2] Bx  = x [:,:,2]
        cdef np.ndarray[double, ndim=2] By  = x [:,:,3]
        cdef np.ndarray[double, ndim=2] P   = x [:,:,4]
        
        cdef np.ndarray[double, ndim=2] Vxh = xh[:,:,0]
        cdef np.ndarray[double, ndim=2] Vyh = xh[:,:,1]
        cdef np.ndarray[double, ndim=2] Bxh = xh[:,:,2]
        cdef np.ndarray[double, ndim=2] Byh = xh[:,:,3]
        cdef np.ndarray[double, ndim=2] Ph  = xh[:,:,4]
        
        cdef double[:,:] Vx_ave = 0.5 * (Vx + Vxh)
        cdef double[:,:] Vy_ave = 0.5 * (Vy + Vyh)
        cdef double[:,:] Bx_ave = 0.5 * (Bx + Bxh)
        cdef double[:,:] By_ave = 0.5 * (By + Byh)
        cdef double[:,:] P_ave  = 0.5 * (P  + Ph )

        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = self.derivatives.dt(Vx,  ix, jx) \
                             - self.derivatives.dt(Vxh, ix, jx) \
                             + 0.5 * self.derivatives.psix(Vx,  Vy,  Vx,  Vy,  ix, jx) \
                             + 0.5 * self.derivatives.psix(Vxh, Vyh, Vxh, Vyh, ix, jx) \
                             - 0.5 * self.derivatives.psix(Bx,  By,  Bx,  By,  ix, jx) \
                             - 0.5 * self.derivatives.psix(Bxh, Byh, Bxh, Byh, ix, jx) \
                             + 1.0 * self.derivatives.divx_sg(P,  ix, jx)
                
                # V_y
                y[iy, jy, 1] = self.derivatives.dt(Vy,  ix, jx) \
                             - self.derivatives.dt(Vyh, ix, jx) \
                             + 0.5 * self.derivatives.psiy(Vx,  Vy,  Vx,  Vy,  ix, jx) \
                             + 0.5 * self.derivatives.psiy(Vxh, Vyh, Vxh, Vyh, ix, jx) \
                             - 0.5 * self.derivatives.psiy(Bx,  By,  Bx,  By,  ix, jx) \
                             - 0.5 * self.derivatives.psiy(Bxh, Byh, Bxh, Byh, ix, jx) \
                             + 1.0 * self.derivatives.divy_sg(P,  ix, jx)
                              
                # B_x
                y[iy, jy, 2] = self.derivatives.dt(Bx,  ix, jx) \
                             - self.derivatives.dt(Bxh, ix, jx) \
                             + self.derivatives.phix(Vxh,  By_ave,  ix, jx) \
                             - self.derivatives.phix(Bx_ave,  Vyh,  ix, jx)
                    
                # B_y
                y[iy, jy, 3] = self.derivatives.dt(By,  ix, jx) \
                             - self.derivatives.dt(Byh, ix, jx) \
                             + self.derivatives.phiy(Vxh,  By_ave,  ix, jx) \
                             - self.derivatives.phiy(Bx_ave,  Vyh,  ix, jx)
                
                # P
                y[iy, jy, 4] = self.derivatives.gradx_simple(Vx, ix, jx) \
                             + self.derivatives.grady_simple(Vy, ix, jx)
                             


    @cython.boundscheck(False)
    def timestep(self, np.ndarray[double, ndim=3] x,
                       np.ndarray[double, ndim=3] y):
        
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double[:,:] Vx = x[:,:,0]
        cdef double[:,:] Vy = x[:,:,1]
        cdef double[:,:] Bx = x[:,:,2]
        cdef double[:,:] By = x[:,:,3]
        cdef double[:,:] P  = x[:,:,4]

        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = \
                             + self.derivatives.psix(Vx, Vy, Vx, Vy, ix, jx) \
                             - self.derivatives.psix(Bx, By, Bx, By, ix, jx) \
                             + self.derivatives.divx_sg(P, ix, jx)
                
                # V_y
                y[iy, jy, 1] = \
                             + self.derivatives.psiy(Vx, Vy, Vx, Vy, ix, jx) \
                             - self.derivatives.psiy(Bx, By, Bx, By, ix, jx) \
                             + self.derivatives.divy_sg(P, ix, jx)
                              
                # B_x
                y[iy, jy, 2] = \
                             + self.derivatives.phix(Vx, By, ix, jx) \
                             - self.derivatives.phix(Bx, Vy, ix, jx)
                    
                # B_y
                y[iy, jy, 3] = \
                             + self.derivatives.phiy(Vx, By, ix, jx) \
                             - self.derivatives.phiy(Bx, Vy, ix, jx)
                
                # P
                y[iy, jy, 4] = P[ix,jx]
                             

    @cython.boundscheck(False)
    def formMat(self, Mat A, Mat P = None):
        cdef np.int64_t i, j, ia, ja, ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da7.globalToLocal(self.Xp, self.localXp)
        self.da7.globalToLocal(self.Xh, self.localXh)
        
        xp = self.da7.getVecArray(self.localXp)[...]
        xh = self.da7.getVecArray(self.localXh)[...]
        
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
                self.psix_ux(A_arr, Vxp, Vyp, ix, jx, +1)
                
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
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - psix_ux(Bx, By )
                
                A_arr = np.zeros((5,5))
                
                self.psix_ux(A_arr, Bxp, Byp, ix, jx, -1)
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - psix_uy(Bx, By)
                
                A_arr = np.zeros((5,5))
                
                self.psix_uy(A_arr, Bxp, Byp, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dx(P)
                
                col.field = 4
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
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(Vy)
                # + psiy_uy(Vx, Vy)
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
                self.psiy_uy(A_arr, Vxp, Vyp, ix, jx, +1)
                
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
                
                col.field = 2
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # - psiy_uy(Bx, By)
                
                A_arr = np.zeros((5,5))
                
                self.psiy_uy(A_arr, Bxp, Byp, ix, jx, -1)
                
                col.field = 3
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dy(P)
                col.field = 4
                for index, value in [
                        ((i,   j  ), + 4. * self.fac_grdy),
                        ((i,   j-1), - 4. * self.fac_grdy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # B_x
                row.index = (i,j)
                row.field = 2
                
                # dt(dBx)
                # - phix(dBx, Vy )
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
                
                
                # + phix(Vx,  dBy)
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
                
                # - phiy(dBx, Vy )
                # - phiy(dBx, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.phiy_ux(A_arr, Vyh, ix, jx, -1)
                
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
    cdef double dt(self, double[:,:] A,
                                 int i, int j):
        
        # (i,   j  )
        A[2,2] += self.ht_inv
        
    
    
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


