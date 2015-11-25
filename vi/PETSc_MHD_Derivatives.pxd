'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np


cdef class PETSc_MHD_Derivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    cdef int  nx
    cdef int  ny
    
    cdef double ht
    cdef double hx
    cdef double hy
    
    cdef double ht_inv
    cdef double hx_inv
    cdef double hy_inv
    
    cdef double hx_inv2
    cdef double hy_inv2
    
    
        
    cdef double dx(self, double[:,:] B,
                               double[:,:] V,
                               int i, int j)
    
    cdef double dy(self, double[:,:] B,
                               double[:,:] V,
                               int i, int j)

    cdef double fdudx(self, double[:,:] F,
                                  double[:,:] U,
                                  int i, int j)

    cdef double fdudy(self, double[:,:] F,
                                  double[:,:] U,
                                  int i, int j)
    
    cdef double fx_dx_ux(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j)
    
    cdef double fx_dx_uy(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j)
    
    cdef double fy_dy_ux(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j)
    
    cdef double fy_dy_uy(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j)
    
    cdef double dx_fx_uy(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j)
    
    cdef double dy_fx_uy(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j)
    

    cdef double gradx(self, double[:,:] x,
                                  int i, int j)
    
    cdef double grady(self, double[:,:] x,
                                  int i, int j)
    
    cdef double gradx_sg(self, double[:,:] x,
                                     int i, int j)

    cdef double grady_sg(self, double[:,:] x,
                                     int i, int j)

    cdef double gradx_fv(self, double[:,:] x,
                                     int i, int j)
    
    cdef double grady_fv(self, double[:,:] x,
                                     int i, int j)

    cdef double gradx_simple(self, double[:,:] x,
                                         int i, int j)
    
    cdef double grady_simple(self, double[:,:] x,
                                         int i, int j)

    cdef double divx(self, double[:,:] x,
                                 int i, int j)
    
    cdef double divy(self, double[:,:] x,
                                 int i, int j)

    cdef double divx_sg(self, double[:,:] x,
                                    int i, int j)

    cdef double divy_sg(self, double[:,:] x,
                                    int i, int j)

    cdef double laplace(self, double[:,:] x,
                                    int i, int j)

    cdef double laplace_fd(self, double[:,:] x,
                                       int i, int j)


    cdef double dt(self, double[:,:] x,
                               int i, int j)
    
    cdef double dt_diag(self, double[:,:] x,
                                    int i, int j)


    cdef double rot(self, double[:,:] Ux, double[:,:] Uy, int i, int j)

    cdef double psix(self, double[:,:] Ux, double[:,:] Uy, double[:,:] Vx, double[:,:] Vy, int i, int j)
    cdef double psiy(self, double[:,:] Ux, double[:,:] Uy, double[:,:] Vx, double[:,:] Vy, int i, int j)

    cdef double phix(self, double[:,:] F, double[:,:] U, int i, int j)
    cdef double phiy(self, double[:,:] F, double[:,:] U, int i, int j)

