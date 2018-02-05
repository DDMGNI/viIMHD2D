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



cdef class PETScSolverFaraday(object):
    '''
    
    '''
    
    def __init__(self, object da2, object da7,
                 int nx, int ny,
                 double ht, double hx, double hy,
                 double mu, double nu, double eta, double de):
        '''
        Constructor
        '''
        
        # distributed array
        self.da2 = da2
        self.da7 = da7
        
        # grid size
        self.nx = nx
        self.ny = ny
        
        # step size
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        # factors of derivatives
        self.ht_inv = 1. / self.ht
        self.fac_dt = 1.0 / 4. / self.ht
        
        self.fac_grdx  = 1.0 / 4. / self.hx
        self.fac_grdy  = 1.0 / 4. / self.hy
        
        self.fac_divx  = 0.5 / self.hx
        self.fac_divy  = 0.5 / self.hy
        
        
        # friction, viscosity, resistivity, electron skin depth
        self.mu  = mu
        self.nu  = nu
        self.eta = eta
        self.de  = de
        
        # create history vectors
        self.Bi = da2.createGlobalVec()
        self.Xh = da7.createGlobalVec()
        self.Xp = da7.createGlobalVec()
        
        # create local vectors
        self.localBi = da2.createLocalVec()
        self.localX  = da7.createLocalVec()
        self.localXh = da7.createLocalVec()
        self.localXp = da7.createLocalVec()
        
        # create derivatives object
        self.derivatives = MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
    
    def update_previous_Bi(self, Vec Bi):
        Bi.copy(self.Bi)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        # copy solution from B to x vector
        x_arr = self.da7.getVecArray(self.Xp)
        b_arr = self.da2.getVecArray(self.Bi)
        
        x_arr[xs:xe, ys:ye, 4] = b_arr[xs:xe, ys:ye, 0]
        x_arr[xs:xe, ys:ye, 5] = b_arr[xs:xe, ys:ye, 1]
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec Bi, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        self.da7.globalToLocal(self.Xp, self.localXp)
        self.da7.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[double, ndim=3] xp = self.da7.getVecArray(self.localXp)[...]
        cdef np.ndarray[double, ndim=3] xh = self.da7.getVecArray(self.localXh)[...]

        self.da2.globalToLocal(Bi, self.localBi)
        
        cdef np.ndarray[double, ndim=3] b = self.da2.getVecArray(self.localBi)[...]
        cdef np.ndarray[double, ndim=3] y = self.da2.getVecArray(Y)[...]

        cdef np.ndarray[double, ndim=2] Bix  = b [:,:,0]
        cdef np.ndarray[double, ndim=2] Biy  = b [:,:,1]
        
        cdef np.ndarray[double, ndim=2] Vxp  = xp[:,:,0]
        cdef np.ndarray[double, ndim=2] Vyp  = xp[:,:,1]
        cdef np.ndarray[double, ndim=2] Bxp  = xp[:,:,2]
        cdef np.ndarray[double, ndim=2] Byp  = xp[:,:,3]
        
        cdef np.ndarray[double, ndim=2] Vxh  = xh[:,:,0]
        cdef np.ndarray[double, ndim=2] Vyh  = xh[:,:,1]
        cdef np.ndarray[double, ndim=2] Bxh  = xh[:,:,2]
        cdef np.ndarray[double, ndim=2] Byh  = xh[:,:,3]
        cdef np.ndarray[double, ndim=2] Bixh = xh[:,:,4]
        cdef np.ndarray[double, ndim=2] Biyh = xh[:,:,5]
        
        cdef double[:,:] Vx_ave  = 0.5 * (Vxp + Vxh )
        cdef double[:,:] Vy_ave  = 0.5 * (Vyp + Vyh )
        cdef double[:,:] Bx_ave  = 0.5 * (Bxp + Bxh )
        cdef double[:,:] By_ave  = 0.5 * (Byp + Byh )
        cdef double[:,:] Bix_ave = 0.5 * (Bix + Bixh)
        cdef double[:,:] Biy_ave = 0.5 * (Biy + Biyh)

        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # Bi_x
                y[iy, jy, 0] = self.derivatives.dt(Bix,  ix, jx) \
                             - self.derivatives.dt(Bixh, ix, jx) \
                             + self.derivatives.phix(Vx_ave,  Biy_ave, ix, jx) \
                             - self.derivatives.phix(Bix_ave, Vy_ave,  ix, jx)
                
                # Bi_y
                y[iy, jy, 1] = self.derivatives.dt(Biy,  ix, jx) \
                             - self.derivatives.dt(Biyh, ix, jx) \
                             + self.derivatives.phiy(Vx_ave,  Biy_ave, ix, jx) \
                             - self.derivatives.phiy(Bix_ave, Vy_ave,  ix, jx)

                             


    @cython.boundscheck(False)
    def formMat(self, Mat A, Mat P = None):
        cdef np.int64_t i, j, ia, ja, ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        self.da7.globalToLocal(self.Xp, self.localXp)
        self.da7.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[double, ndim=3] xp = self.da7.getVecArray(self.localXp)[...]
        cdef np.ndarray[double, ndim=3] xh = self.da7.getVecArray(self.localXh)[...]
        
        cdef np.ndarray[double, ndim=2] Vxp  = xp[:,:,0]
        cdef np.ndarray[double, ndim=2] Vyp  = xp[:,:,1]
        cdef np.ndarray[double, ndim=2] Bxp  = xp[:,:,2]
        cdef np.ndarray[double, ndim=2] Byp  = xp[:,:,3]
        cdef np.ndarray[double, ndim=2] Bixp = xp[:,:,4]
        cdef np.ndarray[double, ndim=2] Biyp = xp[:,:,5]
        
        cdef np.ndarray[double, ndim=2] Vxh  = xh[:,:,0]
        cdef np.ndarray[double, ndim=2] Vyh  = xh[:,:,1]
        cdef np.ndarray[double, ndim=2] Bxh  = xh[:,:,2]
        cdef np.ndarray[double, ndim=2] Byh  = xh[:,:,3]
        cdef np.ndarray[double, ndim=2] Bixh = xh[:,:,4]
        cdef np.ndarray[double, ndim=2] Biyh = xh[:,:,5]
        
        cdef double[:,:] Vx_ave  = 0.5 * (Vxp  + Vxh )
        cdef double[:,:] Vy_ave  = 0.5 * (Vyp  + Vyh )
        cdef double[:,:] Bx_ave  = 0.5 * (Bxp  + Bxh )
        cdef double[:,:] By_ave  = 0.5 * (Byp  + Byh )
        cdef double[:,:] Bix_ave = 0.5 * (Bixp + Bixh)
        cdef double[:,:] Biy_ave = 0.5 * (Biyp + Biyh)

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
                
                # Bi_x
                row.index = (i,j)
                row.field = 0

                # dt(dBx)
                # - phix(dBix, Vy )
                # - phix(dBix, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
                self.phix_ux(A_arr, Vy_ave, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # + phix(Vx,  dBiy)
                # + phix(Vxh, dBiy)
                
                A_arr = np.zeros((5,5))
                
                self.phix_uy(A_arr, Vx_ave, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # Bi_y
                row.index = (i,j)
                row.field = 1
                
                # - phiy(dBix, Vy )
                # - phiy(dBix, Vyh)
                
                A_arr = np.zeros((5,5))
                
                self.phiy_ux(A_arr, Vy_ave, ix, jx, -1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(dBiy)
                # + phiy(Vx,  dBiy)
                # + phiy(Vxh, dBiy)

                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
                self.phiy_uy(A_arr, Vx_ave, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])

                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
        A.assemble()
        
        if P != None:
            P.assemble()
        
                
    

    @cython.boundscheck(False)
    cdef double muu(self, double[:,:] A,
                                 int i, int j):
        
        # (i,   j  )
        A[2,2] += 0.5 * self.mu
        
    
    
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
        A[2, 3] -= ( Fy[i-1, j+1] + Fy[i, j+1] ) * fac
        
        

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


        