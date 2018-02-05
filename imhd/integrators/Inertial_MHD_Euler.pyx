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



cdef class PETScSolverEuler(object):
    '''
    
    '''
    
    def __init__(self, object da3, object da7,
                 int nx, int ny,
                 double ht, double hx, double hy,
                 double mu, double nu, double eta, double de):
        '''
        Constructor
        '''
        
        # distributed array
        self.da3 = da3
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
        self.Vp = da3.createGlobalVec()
        self.Xh = da7.createGlobalVec()
        self.Xp = da7.createGlobalVec()
        
        # create local vectors
        self.localV  = da3.createLocalVec()
        self.localX  = da7.createLocalVec()
        self.localXh = da7.createLocalVec()
        self.localXp = da7.createLocalVec()
        
        # create derivatives object
        self.derivatives = MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
    
    def update_previous_V(self, Vec V):
        V.copy(self.Vp)
        
        (xs, xe), (ys, ye) = self.da3.getRanges()
        
        # copy solution from B to x vector
        x_arr = self.da7.getVecArray(self.Xp)
        v_arr = self.da3.getVecArray(self.Vp)
        
        x_arr[xs:xe, ys:ye, 0] = v_arr[xs:xe, ys:ye, 0]
        x_arr[xs:xe, ys:ye, 1] = v_arr[xs:xe, ys:ye, 1]
        x_arr[xs:xe, ys:ye, 6] = v_arr[xs:xe, ys:ye, 2]
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec V, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da3.getRanges()
        
        self.da7.globalToLocal(self.Xp, self.localXp)
        self.da7.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[double, ndim=3] xp = self.da7.getVecArray(self.localXp)[...]
        cdef np.ndarray[double, ndim=3] xh = self.da7.getVecArray(self.localXh)[...]

        self.da3.globalToLocal(V, self.localV)
        
        cdef np.ndarray[double, ndim=3] v = self.da3.getVecArray(self.localV)[...]
        cdef np.ndarray[double, ndim=3] y = self.da3.getVecArray(Y)[...]

        cdef np.ndarray[double, ndim=2] Vx  =  v [:,:,0]
        cdef np.ndarray[double, ndim=2] Vy   = v [:,:,1]
        cdef np.ndarray[double, ndim=2] P    = v [:,:,2]
        
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
        cdef np.ndarray[double, ndim=2] Ph   = xh[:,:,6]
        
        cdef double[:,:] Vx_ave  = 0.5 * (Vx   + Vxh )
        cdef double[:,:] Vy_ave  = 0.5 * (Vy   + Vyh )
        cdef double[:,:] Bx_ave  = 0.5 * (Bxp  + Bxh )
        cdef double[:,:] By_ave  = 0.5 * (Byp  + Byh )
        cdef double[:,:] Bix_ave = 0.5 * (Bixp + Bixh)
        cdef double[:,:] Biy_ave = 0.5 * (Biyp + Biyh)
        cdef double[:,:] P_ave   = 0.5 * (P    + Ph  )

        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # V_x
                y[iy, jy, 0] = self.derivatives.dt(Vx,  ix, jx) \
                             - self.derivatives.dt(Vxh, ix, jx) \
                             + self.derivatives.psix(Vx_ave,  Vy_ave,  Vx_ave, Vy_ave, ix, jx) \
                             - self.derivatives.psix(Bix_ave, Biy_ave, Bx_ave, By_ave, ix, jx) \
                             + self.derivatives.divx_sg(P,  ix, jx) \
                             + 0.5 * self.mu * Vx [ix,jx] \
                             + 0.5 * self.mu * Vxh[ix,jx]
                
                # V_y
                y[iy, jy, 1] = self.derivatives.dt(Vy,  ix, jx) \
                             - self.derivatives.dt(Vyh, ix, jx) \
                             + self.derivatives.psiy(Vx_ave,  Vy_ave,  Vx_ave, Vy_ave, ix, jx) \
                             - self.derivatives.psiy(Bix_ave, Biy_ave, Bx_ave, By_ave, ix, jx) \
                             + self.derivatives.divy_sg(P,  ix, jx) \
                             + 0.5 * self.mu * Vy [ix,jx] \
                             + 0.5 * self.mu * Vyh[ix,jx]
                
                # P
                y[iy, jy, 2] = self.derivatives.gradx_simple(Vx_ave, ix, jx) \
                             + self.derivatives.grady_simple(Vy_ave, ix, jx)
                


    @cython.boundscheck(False)
    def formMat(self, Mat A, Mat P = None):
        cdef int i, j, ia, ja, ix, jx
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da3.getRanges()
        
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
        cdef np.ndarray[double, ndim=2] Pp   = xp[:,:,6]
        
        cdef np.ndarray[double, ndim=2] Vxh  = xh[:,:,0]
        cdef np.ndarray[double, ndim=2] Vyh  = xh[:,:,1]
        cdef np.ndarray[double, ndim=2] Bxh  = xh[:,:,2]
        cdef np.ndarray[double, ndim=2] Byh  = xh[:,:,3]
        cdef np.ndarray[double, ndim=2] Bixh = xh[:,:,4]
        cdef np.ndarray[double, ndim=2] Biyh = xh[:,:,5]
        cdef np.ndarray[double, ndim=2] Ph   = xh[:,:,6]
        
        cdef double[:,:] Vx_ave  = 0.5 * (Vxp  + Vxh )
        cdef double[:,:] Vy_ave  = 0.5 * (Vyp  + Vyh )
        cdef double[:,:] Bx_ave  = 0.5 * (Bxp  + Bxh )
        cdef double[:,:] By_ave  = 0.5 * (Byp  + Byh )
        cdef double[:,:] Bix_ave = 0.5 * (Bixp + Bixh)
        cdef double[:,:] Biy_ave = 0.5 * (Biyp + Biyh)
        cdef double[:,:] P_ave   = 0.5 * (Pp   + Ph  )

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
                # + psix_ux(dVx, dVy, Vx, Vy)
                # + psix_vx(Vx, Vy, dVx, dVy)
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
                self.psix_ux(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                self.psix_vx(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                self.muu(A_arr, ix, jx)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                
                # + psix_uy(dVx, dVy, Vx, Vy)
                # + psix_vy(Vx, Vy, dVx, dVy)
                
                A_arr = np.zeros((5,5))
                
                self.psix_uy(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                self.psix_vy(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                # + dx(P)
                
                col.field = 2
                for index, value in [
                        ((i,   j  ), + 4. * self.fac_grdx),
                        ((i-1, j  ), - 4. * self.fac_grdx),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # V_y
                row.index = (i,j)
                row.field = 1
                
                # + psiy_ux(dVx, dVy, Vx, Vy)
                # + psiy_vx(Vx, Vy, dVx, dVy)
                
                A_arr = np.zeros((5,5))
                
                self.psiy_ux(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                self.psiy_vx(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                
                col.field = 0
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                
                # dt(Vy)
                # + psiy_uy(dVx, dVy, Vx, Vy)
                # + psiy_vy(Vx, Vy, dVx, dVy)
                
                A_arr = np.zeros((5,5))
                
                self.dt(A_arr, ix, jx)
                self.psiy_uy(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                self.psiy_vy(A_arr, Vx_ave, Vy_ave, ix, jx, +1)
                self.muu(A_arr, ix, jx)
                
                col.field = 1
                for ia in range(0,5):
                    for ja in range(0,5):
                        col.index = (i-2+ia, j-2+ja)
                        A.setValueStencil(row, col, A_arr[ia,ja])
                
                if P != None:
                    P.setValueStencil(row, row, 1. / A_arr[2,2])
                
                # + dy(P)
                col.field = 2
                for index, value in [
                        ((i,   j  ), + 4. * self.fac_grdy),
                        ((i,   j-1), - 4. * self.fac_grdy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # P
                row.index = (i,j)
                row.field = 2
                
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



    cdef double psix_ux(self, double[:,:] A,
                                    double[:,:] Vx,
                                    double[:,:] Vy,
                                    int i, int j,
                                    double sign):
        
        pass
        
        
    cdef double psix_vx(self, double[:,:] A,
                                    double[:,:] Ux,
                                    double[:,:] Uy,
                                    int i, int j,
                                    double sign):
        cdef double fac = sign * 0.25 * 0.5 / self.hy
        
        # Ux[i,   j-1]
        A[2, 1] -= ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac
        
        # Ux[i,   j  ]
        A[2, 2] += ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac
        A[2, 2] -= ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac
        
        # Ux[i,   j+1]
        A[2, 3] += ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac
        
        
    cdef double psix_uy(self, double[:,:] A,
                                    double[:,:] Vx,
                                    double[:,:] Vy,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5
        
        # Uy[i-1, j  ]
        A[1, 2] -= self.rot(Vx, Vy, i,   j  ) * fac

        # Uy[i-1, j+1]
        A[1, 3] -= self.rot(Vx, Vy, i,   j+1) * fac 
        
 
        # Uy[i,   j  ]
        A[2, 2] -= self.rot(Vx, Vy, i,   j  ) * fac
        
        # Uy[i,   j+1]
        A[2, 3] -= self.rot(Vx, Vy, i,   j+1) * fac 
        

    cdef double psix_vy(self, double[:,:] A,
                                    double[:,:] Ux,
                                    double[:,:] Uy,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5 / self.hx
        
        # Uy[i-1, j  ]
        A[1, 2] += ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac

        # Uy[i-1, j+1]
        A[1, 3] += ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac
        
 
        # Uy[i,   j  ]
        A[2, 2] -= ( Uy[i-1, j  ] + Uy[i,   j  ] ) * fac
        
        # Uy[i,   j+1]
        A[2, 3] -= ( Uy[i-1, j+1] + Uy[i,   j+1] ) * fac
        

    cdef double psiy_ux(self, double[:,:] A,
                                    double[:,:] Vx,
                                    double[:,:] Vy,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5

        # Ux[i,   j-1]
        A[2, 1] += self.rot(Vx, Vy, i,   j  ) * fac

        # Ux[i,   j  ]
        A[2, 2] += self.rot(Vx, Vy, i,   j  ) * fac 
        
        
        # Ux[i+1, j-1]
        A[3, 1] += self.rot(Vx, Vy, i+1, j  ) * fac 

        # Ux[i+1, j  ]
        A[3, 2] += self.rot(Vx, Vy, i+1, j  ) * fac 


    cdef double psiy_vx(self, double[:,:] A,
                                    double[:,:] Ux,
                                    double[:,:] Uy,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5 / self.hy

        # Ux[i,   j-1]
        A[2, 1] += ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac

        # Ux[i,   j  ]
        A[2, 2] -= ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac
        
        
        # Ux[i+1, j-1]
        A[3, 1] += ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac

        # Ux[i+1, j  ]
        A[3, 2] -= ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac


    cdef double psiy_uy(self, double[:,:] A,
                                    double[:,:] Vx,
                                    double[:,:] Vy,
                                    int i, int j,
                                    double sign):
        pass
        


    cdef double psiy_vy(self, double[:,:] A,
                                    double[:,:] Ux,
                                    double[:,:] Uy,
                                    int i, int j,
                                    double sign):

        cdef double fac = sign * 0.25 * 0.5 / self.hx
        
        # Uy[i-1, j  ]
        A[1, 2] -= ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac

        # Ux[i,   j  ]
        A[2, 2] += ( Ux[i,   j-1] + Ux[i,   j  ] ) * fac
        A[2, 2] -= ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac
        
        
        # Ux[i+1, j  ]
        A[3, 2] += ( Ux[i+1, j-1] + Ux[i+1, j  ] ) * fac

        