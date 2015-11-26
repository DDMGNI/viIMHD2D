'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np


cdef class MHD_Derivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    
    def __cinit__(self,
                  int  nx, int  ny,
                  double ht, double hx, double hy):
        '''
        Constructor
        '''
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        self.ht_inv = 1. / ht
        self.hx_inv = 1. / hx
        self.hy_inv = 1. / hy
        
        self.hx_inv2 = self.hx_inv**2
        self.hy_inv2 = self.hy_inv**2
        
        
        
    @cython.boundscheck(False)
    cdef double dx(self, double[:,:] B,
                               double[:,:] V,
                               int i, int j):
        '''
        MHD Derivative: dx
        '''
        
        cdef double result
        
        result = ( \
                     + 1. * B[i-1, j-1] * V[i-1, j-1] \
                     - 1. * B[i+1, j-1] * V[i+1, j-1] \
                     + 1. * B[i-1, j  ] * V[i-1, j-1] \
                     - 1. * B[i+1, j  ] * V[i+1, j-1] \
                     + 1. * B[i-1, j-1] * V[i-1, j  ] \
                     - 1. * B[i+1, j-1] * V[i+1, j  ] \
                     + 2. * B[i-1, j  ] * V[i-1, j  ] \
                     - 2. * B[i+1, j  ] * V[i+1, j  ] \
                     + 1. * B[i-1, j+1] * V[i-1, j  ] \
                     - 1. * B[i+1, j+1] * V[i+1, j  ] \
                     + 1. * B[i-1, j  ] * V[i-1, j+1] \
                     - 1. * B[i+1, j  ] * V[i+1, j+1] \
                     + 1. * B[i-1, j+1] * V[i-1, j+1] \
                     - 1. * B[i+1, j+1] * V[i+1, j+1] \
                 ) * self.hx_inv / 16.
 
        return result
    
    
    @cython.boundscheck(False)
    cdef double fdudx(self, double[:,:] F,
                                  double[:,:] U,
                                  int i, int j):
        '''
        MHD Derivative: full single derivative F d_x D
        '''
        
        cdef double result
        
        result = ( \
                     - 1  * U[i-1, j-1] * F[i-1, j-1] \
                     - 1  * U[i-1, j-1] * F[i-1, j  ] \
                     - 1  * U[i-1, j-1] * F[i,   j-1] \
                     - 1  * U[i-1, j-1] * F[i,   j  ] \
                     - 1  * U[i-1, j+1] * F[i-1, j+1] \
                     - 1  * U[i-1, j+1] * F[i-1, j  ] \
                     - 1  * U[i-1, j+1] * F[i,   j+1] \
                     - 1  * U[i-1, j+1] * F[i,   j  ] \
                     - 1  * U[i-1, j  ] * F[i-1, j-1] \
                     - 1  * U[i-1, j  ] * F[i-1, j+1] \
                     - 2  * U[i-1, j  ] * F[i-1, j  ] \
                     - 1  * U[i-1, j  ] * F[i,   j-1] \
                     - 1  * U[i-1, j  ] * F[i,   j+1] \
                     - 2  * U[i-1, j  ] * F[i,   j  ] \
                     + 1  * U[i+1, j-1] * F[i+1, j-1] \
                     + 1  * U[i+1, j-1] * F[i+1, j  ] \
                     + 1  * U[i+1, j-1] * F[i,   j-1] \
                     + 1  * U[i+1, j-1] * F[i,   j  ] \
                     + 1  * U[i+1, j+1] * F[i+1, j+1] \
                     + 1  * U[i+1, j+1] * F[i+1, j  ] \
                     + 1  * U[i+1, j+1] * F[i,   j+1] \
                     + 1  * U[i+1, j+1] * F[i,   j  ] \
                     + 1  * U[i+1, j  ] * F[i+1, j-1] \
                     + 1  * U[i+1, j  ] * F[i+1, j+1] \
                     + 2  * U[i+1, j  ] * F[i+1, j  ] \
                     + 1  * U[i+1, j  ] * F[i,   j-1] \
                     + 1  * U[i+1, j  ] * F[i,   j+1] \
                     + 2  * U[i+1, j  ] * F[i,   j  ] \
                     + 1  * U[i,   j-1] * F[i-1, j-1] \
                     + 1  * U[i,   j-1] * F[i-1, j  ] \
                     - 1  * U[i,   j-1] * F[i+1, j-1] \
                     - 1  * U[i,   j-1] * F[i+1, j  ] \
                     + 1  * U[i,   j+1] * F[i-1, j+1] \
                     + 1  * U[i,   j+1] * F[i-1, j  ] \
                     - 1  * U[i,   j+1] * F[i+1, j+1] \
                     - 1  * U[i,   j+1] * F[i+1, j  ] \
                     + 1  * U[i,   j  ] * F[i-1, j-1] \
                     + 1  * U[i,   j  ] * F[i-1, j+1] \
                     + 2  * U[i,   j  ] * F[i-1, j  ] \
                     - 1  * U[i,   j  ] * F[i+1, j-1] \
                     - 1  * U[i,   j  ] * F[i+1, j+1] \
                     - 2  * U[i,   j  ] * F[i+1, j  ] \
                 ) * self.hx_inv / 32.
        
        return result
    
        
    
    @cython.boundscheck(False)
    cdef double fx_dx_ux(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j):
        '''
        MHD Derivative: full single derivative F_x d_x U_x
        '''
        
        cdef double result
        
        result = ( \
                     - 1. * U[i-2, j  ] * F[i-2, j  ] \
                     - 1. * U[i-2, j  ] * F[i-1, j-1] \
                     - 1. * U[i-2, j  ] * F[i-1, j+1] \
                     - 4. * U[i-2, j  ] * F[i-1, j  ] \
                     - 1. * U[i-2, j  ] * F[i,   j  ] \
                     - 1. * U[i-1, j-1] * F[i-1, j-1] \
                     - 1. * U[i-1, j-1] * F[i+1, j-1] \
                     - 1. * U[i-1, j-1] * F[i,   j-2] \
                     - 4. * U[i-1, j-1] * F[i,   j-1] \
                     - 1. * U[i-1, j-1] * F[i,   j  ] \
                     - 1. * U[i-1, j+1] * F[i-1, j+1] \
                     - 1. * U[i-1, j+1] * F[i+1, j+1] \
                     - 4. * U[i-1, j+1] * F[i,   j+1] \
                     - 1. * U[i-1, j+1] * F[i,   j+2] \
                     - 1. * U[i-1, j+1] * F[i,   j  ] \
                     - 4. * U[i-1, j  ] * F[i-1, j  ] \
                     - 4. * U[i-1, j  ] * F[i+1, j  ] \
                     - 4. * U[i-1, j  ] * F[i,   j-1] \
                     - 4. * U[i-1, j  ] * F[i,   j+1] \
                     - 16 * U[i-1, j  ] * F[i,   j  ] \
                     + 1. * U[i+1, j-1] * F[i-1, j-1] \
                     + 1. * U[i+1, j-1] * F[i+1, j-1] \
                     + 1. * U[i+1, j-1] * F[i,   j-2] \
                     + 4. * U[i+1, j-1] * F[i,   j-1] \
                     + 1. * U[i+1, j-1] * F[i,   j  ] \
                     + 1. * U[i+1, j+1] * F[i-1, j+1] \
                     + 1. * U[i+1, j+1] * F[i+1, j+1] \
                     + 4. * U[i+1, j+1] * F[i,   j+1] \
                     + 1. * U[i+1, j+1] * F[i,   j+2] \
                     + 1. * U[i+1, j+1] * F[i,   j  ] \
                     + 4. * U[i+1, j  ] * F[i-1, j  ] \
                     + 4. * U[i+1, j  ] * F[i+1, j  ] \
                     + 4. * U[i+1, j  ] * F[i,   j-1] \
                     + 4. * U[i+1, j  ] * F[i,   j+1] \
                     + 16 * U[i+1, j  ] * F[i,   j  ] \
                     + 1. * U[i+2, j  ] * F[i+1, j-1] \
                     + 1. * U[i+2, j  ] * F[i+1, j+1] \
                     + 4. * U[i+2, j  ] * F[i+1, j  ] \
                     + 1. * U[i+2, j  ] * F[i+2, j  ] \
                     + 1. * U[i+2, j  ] * F[i,   j  ] \
                     + 1. * U[i,   j  ] * F[i-2, j  ] \
                     + 1. * U[i,   j  ] * F[i-1, j-1] \
                     + 1. * U[i,   j  ] * F[i-1, j+1] \
                     + 4. * U[i,   j  ] * F[i-1, j  ] \
                     - 1. * U[i,   j  ] * F[i+1, j-1] \
                     - 1. * U[i,   j  ] * F[i+1, j+1] \
                     - 4. * U[i,   j  ] * F[i+1, j  ] \
                     - 1. * U[i,   j  ] * F[i+2, j  ] \
                 ) * self.hx_inv / 128.
         
        return result
    
        
    
    @cython.boundscheck(False)
    cdef double fx_dx_uy(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j):
        '''
        MHD Derivative: full single derivative F_x d_x U_y
        '''
        
        cdef double result
        
        result = ( \
                     - 2. * U[i-2, j  ] * F[i-2, j+1] \
                     - 2. * U[i-2, j  ] * F[i-2, j  ] \
                     - 2. * U[i-2, j  ] * F[i-1, j+1] \
                     - 2. * U[i-2, j  ] * F[i-1, j  ] \
                     - 2. * U[i-1, j-1] * F[i-1, j-1] \
                     - 2. * U[i-1, j-1] * F[i-1, j  ] \
                     - 2. * U[i-1, j-1] * F[i,   j-1] \
                     - 2. * U[i-1, j-1] * F[i,   j  ] \
                     - 2. * U[i-1, j+1] * F[i-1, j+1] \
                     - 2. * U[i-1, j+1] * F[i-1, j+2] \
                     - 2. * U[i-1, j+1] * F[i,   j+1] \
                     - 2. * U[i-1, j+1] * F[i,   j+2] \
                     - 8. * U[i-1, j  ] * F[i-1, j+1] \
                     - 8. * U[i-1, j  ] * F[i-1, j  ] \
                     - 8. * U[i-1, j  ] * F[i,   j+1] \
                     - 8. * U[i-1, j  ] * F[i,   j  ] \
                     + 2. * U[i+1, j-1] * F[i-1, j-1] \
                     + 2. * U[i+1, j-1] * F[i-1, j  ] \
                     + 2. * U[i+1, j-1] * F[i,   j-1] \
                     + 2. * U[i+1, j-1] * F[i,   j  ] \
                     + 2. * U[i+1, j+1] * F[i-1, j+1] \
                     + 2. * U[i+1, j+1] * F[i-1, j+2] \
                     + 2. * U[i+1, j+1] * F[i,   j+1] \
                     + 2. * U[i+1, j+1] * F[i,   j+2] \
                     + 8. * U[i+1, j  ] * F[i-1, j+1] \
                     + 8. * U[i+1, j  ] * F[i-1, j  ] \
                     + 8. * U[i+1, j  ] * F[i,   j+1] \
                     + 8. * U[i+1, j  ] * F[i,   j  ] \
                     + 2. * U[i+2, j  ] * F[i+1, j+1] \
                     + 2. * U[i+2, j  ] * F[i+1, j  ] \
                     + 2. * U[i+2, j  ] * F[i,   j+1] \
                     + 2. * U[i+2, j  ] * F[i,   j  ] \
                     + 2. * U[i,   j  ] * F[i-2, j+1] \
                     + 2. * U[i,   j  ] * F[i-2, j  ] \
                     + 2. * U[i,   j  ] * F[i-1, j+1] \
                     + 2. * U[i,   j  ] * F[i-1, j  ] \
                     - 2. * U[i,   j  ] * F[i+1, j+1] \
                     - 2. * U[i,   j  ] * F[i+1, j  ] \
                     - 2. * U[i,   j  ] * F[i,   j+1] \
                     - 2. * U[i,   j  ] * F[i,   j  ] \
                 ) * self.hx_inv / 128.
         
        return result
    
        
    
    cdef double dx_fx_uy(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j):
        '''
        MHD Derivative: full single derivative d_x F_x U_y
        '''
        
        cdef double result
        
        result = ( \
                     - 1. * U[i-2, j  ] * F[i-2, j+1] \
                     - 1. * U[i-2, j  ] * F[i-2, j  ] \
                     - 1. * U[i-1, j-1] * F[i-1, j-1] \
                     - 1. * U[i-1, j-1] * F[i-1, j  ] \
                     - 1. * U[i-1, j+1] * F[i-1, j+1] \
                     - 1. * U[i-1, j+1] * F[i-1, j+2] \
                     - 1. * U[i-1, j  ] * F[i-2, j+1] \
                     - 1. * U[i-1, j  ] * F[i-2, j  ] \
                     - 3. * U[i-1, j  ] * F[i-1, j+1] \
                     - 3. * U[i-1, j  ] * F[i-1, j  ] \
                     + 1. * U[i+1, j-1] * F[i,   j-1] \
                     + 1. * U[i+1, j-1] * F[i,   j  ] \
                     + 1. * U[i+1, j+1] * F[i,   j+1] \
                     + 1. * U[i+1, j+1] * F[i,   j+2] \
                     + 1. * U[i+1, j  ] * F[i+1, j+1] \
                     + 1. * U[i+1, j  ] * F[i+1, j  ] \
                     + 3. * U[i+1, j  ] * F[i,   j+1] \
                     + 3. * U[i+1, j  ] * F[i,   j  ] \
                     + 1. * U[i+2, j  ] * F[i+1, j+1] \
                     + 1. * U[i+2, j  ] * F[i+1, j  ] \
                     - 1. * U[i,   j-1] * F[i-1, j-1] \
                     - 1. * U[i,   j-1] * F[i-1, j  ] \
                     + 1. * U[i,   j-1] * F[i,   j-1] \
                     + 1. * U[i,   j-1] * F[i,   j  ] \
                     - 1. * U[i,   j+1] * F[i-1, j+1] \
                     - 1. * U[i,   j+1] * F[i-1, j+2] \
                     + 1. * U[i,   j+1] * F[i,   j+1] \
                     + 1. * U[i,   j+1] * F[i,   j+2] \
                     - 3. * U[i,   j  ] * F[i-1, j+1] \
                     - 3. * U[i,   j  ] * F[i-1, j  ] \
                     + 3. * U[i,   j  ] * F[i,   j+1] \
                     + 3. * U[i,   j  ] * F[i,   j  ] \
                 ) * self.hx_inv / 32.
         
        return result
                                     
    
    
    @cython.boundscheck(False)
    cdef double dy(self, double[:,:] B,
                               double[:,:] V,
                               int i, int j):
        '''
        MHD Derivative: dy
        '''
        
        cdef double result
        
        result = ( \
                     + 1. * B[i-1, j-1] * V[i-1, j-1] \
                     - 1. * B[i-1, j+1] * V[i-1, j+1] \
                     + 1. * B[i-1, j-1] * V[i,   j-1] \
                     - 1. * B[i-1, j+1] * V[i,   j+1] \
                     + 1. * B[i,   j-1] * V[i-1, j-1] \
                     - 1. * B[i,   j+1] * V[i-1, j+1] \
                     + 1. * B[i,   j-1] * V[i+1, j-1] \
                     - 1. * B[i,   j+1] * V[i+1, j+1] \
                     + 2. * B[i,   j-1] * V[i,   j-1] \
                     - 2. * B[i,   j+1] * V[i,   j+1] \
                     + 1. * B[i+1, j-1] * V[i,   j-1] \
                     - 1. * B[i+1, j+1] * V[i,   j+1] \
                     + 1. * B[i+1, j-1] * V[i+1, j-1] \
                     - 1. * B[i+1, j+1] * V[i+1, j+1] \
                 ) * self.hy_inv / 16.
                 
        return result
    
    
    @cython.boundscheck(False)
    cdef double fdudy(self, double[:,:] F,
                                  double[:,:] U,
                                  int i, int j):
        '''
        MHD Derivative: full single derivative F d_y D
        '''
        
        cdef double result
        
        result = ( \
                     - 1  * U[i-1, j-1] * F[i-1, j-1] \
                     - 1  * U[i-1, j-1] * F[i-1, j  ] \
                     - 1  * U[i-1, j-1] * F[i,   j-1] \
                     - 1  * U[i-1, j-1] * F[i,   j  ] \
                     + 1  * U[i-1, j+1] * F[i-1, j+1] \
                     + 1  * U[i-1, j+1] * F[i-1, j  ] \
                     + 1  * U[i-1, j+1] * F[i,   j+1] \
                     + 1  * U[i-1, j+1] * F[i,   j  ] \
                     + 1  * U[i-1, j  ] * F[i-1, j-1] \
                     - 1  * U[i-1, j  ] * F[i-1, j+1] \
                     + 1  * U[i-1, j  ] * F[i,   j-1] \
                     - 1  * U[i-1, j  ] * F[i,   j+1] \
                     - 1  * U[i+1, j-1] * F[i+1, j-1] \
                     - 1  * U[i+1, j-1] * F[i+1, j  ] \
                     - 1  * U[i+1, j-1] * F[i,   j-1] \
                     - 1  * U[i+1, j-1] * F[i,   j  ] \
                     + 1  * U[i+1, j+1] * F[i+1, j+1] \
                     + 1  * U[i+1, j+1] * F[i+1, j  ] \
                     + 1  * U[i+1, j+1] * F[i,   j+1] \
                     + 1  * U[i+1, j+1] * F[i,   j  ] \
                     + 1  * U[i+1, j  ] * F[i+1, j-1] \
                     - 1  * U[i+1, j  ] * F[i+1, j+1] \
                     + 1  * U[i+1, j  ] * F[i,   j-1] \
                     - 1  * U[i+1, j  ] * F[i,   j+1] \
                     - 1  * U[i,   j-1] * F[i-1, j-1] \
                     - 1  * U[i,   j-1] * F[i-1, j  ] \
                     - 1  * U[i,   j-1] * F[i+1, j-1] \
                     - 1  * U[i,   j-1] * F[i+1, j  ] \
                     - 2  * U[i,   j-1] * F[i,   j-1] \
                     - 2  * U[i,   j-1] * F[i,   j  ] \
                     + 1  * U[i,   j+1] * F[i-1, j+1] \
                     + 1  * U[i,   j+1] * F[i-1, j  ] \
                     + 1  * U[i,   j+1] * F[i+1, j+1] \
                     + 1  * U[i,   j+1] * F[i+1, j  ] \
                     + 2  * U[i,   j+1] * F[i,   j+1] \
                     + 2  * U[i,   j+1] * F[i,   j  ] \
                     + 1  * U[i,   j  ] * F[i-1, j-1] \
                     - 1  * U[i,   j  ] * F[i-1, j+1] \
                     + 1  * U[i,   j  ] * F[i+1, j-1] \
                     - 1  * U[i,   j  ] * F[i+1, j+1] \
                     + 2  * U[i,   j  ] * F[i,   j-1] \
                     - 2  * U[i,   j  ] * F[i,   j+1] \
                  ) * self.hy_inv / 32.
        
        return result
    
    
    
    
    @cython.boundscheck(False)
    cdef double fy_dy_ux(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j):
        '''
        MHD Derivative: full single derivative F_y d_y U_x
        '''
        
        cdef double result
        
        result = ( \
                     - 2. * U[i-1, j-1] * F[i-1, j-1] \
                     - 2. * U[i-1, j-1] * F[i-1, j  ] \
                     - 2. * U[i-1, j-1] * F[i,   j-1] \
                     - 2. * U[i-1, j-1] * F[i,   j  ] \
                     + 2. * U[i-1, j+1] * F[i-1, j-1] \
                     + 2. * U[i-1, j+1] * F[i-1, j  ] \
                     + 2. * U[i-1, j+1] * F[i,   j-1] \
                     + 2. * U[i-1, j+1] * F[i,   j  ] \
                     - 2. * U[i+1, j-1] * F[i+1, j-1] \
                     - 2. * U[i+1, j-1] * F[i+1, j  ] \
                     - 2. * U[i+1, j-1] * F[i+2, j-1] \
                     - 2. * U[i+1, j-1] * F[i+2, j  ] \
                     + 2. * U[i+1, j+1] * F[i+1, j-1] \
                     + 2. * U[i+1, j+1] * F[i+1, j  ] \
                     + 2. * U[i+1, j+1] * F[i+2, j-1] \
                     + 2. * U[i+1, j+1] * F[i+2, j  ] \
                     - 2. * U[i,   j-2] * F[i+1, j-2] \
                     - 2. * U[i,   j-2] * F[i+1, j-1] \
                     - 2. * U[i,   j-2] * F[i,   j-2] \
                     - 2. * U[i,   j-2] * F[i,   j-1] \
                     - 8. * U[i,   j-1] * F[i+1, j-1] \
                     - 8. * U[i,   j-1] * F[i+1, j  ] \
                     - 8. * U[i,   j-1] * F[i,   j-1] \
                     - 8. * U[i,   j-1] * F[i,   j  ] \
                     + 8. * U[i,   j+1] * F[i+1, j-1] \
                     + 8. * U[i,   j+1] * F[i+1, j  ] \
                     + 8. * U[i,   j+1] * F[i,   j-1] \
                     + 8. * U[i,   j+1] * F[i,   j  ] \
                     + 2. * U[i,   j+2] * F[i+1, j+1] \
                     + 2. * U[i,   j+2] * F[i+1, j  ] \
                     + 2. * U[i,   j+2] * F[i,   j+1] \
                     + 2. * U[i,   j+2] * F[i,   j  ] \
                     + 2. * U[i,   j  ] * F[i+1, j-2] \
                     + 2. * U[i,   j  ] * F[i+1, j-1] \
                     - 2. * U[i,   j  ] * F[i+1, j+1] \
                     - 2. * U[i,   j  ] * F[i+1, j  ] \
                     + 2. * U[i,   j  ] * F[i,   j-2] \
                     + 2. * U[i,   j  ] * F[i,   j-1] \
                     - 2. * U[i,   j  ] * F[i,   j+1] \
                     - 2. * U[i,   j  ] * F[i,   j  ] \
                 ) * self.hy_inv / 128.
         
        return result
    
        
    
    @cython.boundscheck(False)
    cdef double fy_dy_uy(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j):
        '''
        MHD Derivative: full single derivative F_y d_y U_y
        '''
        
        cdef double result
        
        result = ( \
                     - 1. * U[i-1, j-1] * F[i-2, j  ] \
                     - 1. * U[i-1, j-1] * F[i-1, j-1] \
                     - 1. * U[i-1, j-1] * F[i-1, j+1] \
                     - 4. * U[i-1, j-1] * F[i-1, j  ] \
                     - 1. * U[i-1, j-1] * F[i,   j  ] \
                     + 1. * U[i-1, j+1] * F[i-2, j  ] \
                     + 1. * U[i-1, j+1] * F[i-1, j-1] \
                     + 1. * U[i-1, j+1] * F[i-1, j+1] \
                     + 4. * U[i-1, j+1] * F[i-1, j  ] \
                     + 1. * U[i-1, j+1] * F[i,   j  ] \
                     - 1. * U[i+1, j-1] * F[i+1, j-1] \
                     - 1. * U[i+1, j-1] * F[i+1, j+1] \
                     - 4. * U[i+1, j-1] * F[i+1, j  ] \
                     - 1. * U[i+1, j-1] * F[i+2, j  ] \
                     - 1. * U[i+1, j-1] * F[i,   j  ] \
                     + 1. * U[i+1, j+1] * F[i+1, j-1] \
                     + 1. * U[i+1, j+1] * F[i+1, j+1] \
                     + 4. * U[i+1, j+1] * F[i+1, j  ] \
                     + 1. * U[i+1, j+1] * F[i+2, j  ] \
                     + 1. * U[i+1, j+1] * F[i,   j  ] \
                     - 1. * U[i,   j-2] * F[i-1, j-1] \
                     - 1. * U[i,   j-2] * F[i+1, j-1] \
                     - 1. * U[i,   j-2] * F[i,   j-2] \
                     - 4. * U[i,   j-2] * F[i,   j-1] \
                     - 1. * U[i,   j-2] * F[i,   j  ] \
                     - 4. * U[i,   j-1] * F[i-1, j  ] \
                     - 4. * U[i,   j-1] * F[i+1, j  ] \
                     - 4. * U[i,   j-1] * F[i,   j-1] \
                     - 4. * U[i,   j-1] * F[i,   j+1] \
                     - 16 * U[i,   j-1] * F[i,   j  ] \
                     + 4. * U[i,   j+1] * F[i-1, j  ] \
                     + 4. * U[i,   j+1] * F[i+1, j  ] \
                     + 4. * U[i,   j+1] * F[i,   j-1] \
                     + 4. * U[i,   j+1] * F[i,   j+1] \
                     + 16 * U[i,   j+1] * F[i,   j  ] \
                     + 1. * U[i,   j+2] * F[i-1, j+1] \
                     + 1. * U[i,   j+2] * F[i+1, j+1] \
                     + 4. * U[i,   j+2] * F[i,   j+1] \
                     + 1. * U[i,   j+2] * F[i,   j+2] \
                     + 1. * U[i,   j+2] * F[i,   j  ] \
                     + 1. * U[i,   j  ] * F[i-1, j-1] \
                     - 1. * U[i,   j  ] * F[i-1, j+1] \
                     + 1. * U[i,   j  ] * F[i+1, j-1] \
                     - 1. * U[i,   j  ] * F[i+1, j+1] \
                     + 1. * U[i,   j  ] * F[i,   j-2] \
                     + 4. * U[i,   j  ] * F[i,   j-1] \
                     - 4. * U[i,   j  ] * F[i,   j+1] \
                     - 1. * U[i,   j  ] * F[i,   j+2] \
                 ) * self.hy_inv / 128.
         
        return result
    
        
    
    cdef double dy_fx_uy(self, double[:,:] F,
                                     double[:,:] U,
                                     int i, int j):
        '''
        MHD Derivative: full single derivative d_y F_x U_y
        '''
        
        cdef double result
        
        result = ( \
                     - 1. * U[i-1, j-1] * F[i-1, j-1] \
                     - 1. * U[i-1, j-1] * F[i-1, j  ] \
                     + 1. * U[i-1, j  ] * F[i-1, j+1] \
                     + 1. * U[i-1, j  ] * F[i-1, j  ] \
                     - 1. * U[i+1, j-2] * F[i,   j-2] \
                     - 1. * U[i+1, j-2] * F[i,   j-1] \
                     - 1. * U[i+1, j-1] * F[i+1, j-1] \
                     - 1. * U[i+1, j-1] * F[i+1, j  ] \
                     - 3. * U[i+1, j-1] * F[i,   j-1] \
                     - 3. * U[i+1, j-1] * F[i,   j  ] \
                     + 1. * U[i+1, j+1] * F[i,   j+1] \
                     + 1. * U[i+1, j+1] * F[i,   j+2] \
                     + 1. * U[i+1, j  ] * F[i+1, j+1] \
                     + 1. * U[i+1, j  ] * F[i+1, j  ] \
                     + 3. * U[i+1, j  ] * F[i,   j+1] \
                     + 3. * U[i+1, j  ] * F[i,   j  ] \
                     - 1. * U[i+2, j-1] * F[i+1, j-1] \
                     - 1. * U[i+2, j-1] * F[i+1, j  ] \
                     + 1. * U[i+2, j  ] * F[i+1, j+1] \
                     + 1. * U[i+2, j  ] * F[i+1, j  ] \
                     - 1. * U[i,   j-2] * F[i,   j-2] \
                     - 1. * U[i,   j-2] * F[i,   j-1] \
                     - 1. * U[i,   j-1] * F[i-1, j-1] \
                     - 1. * U[i,   j-1] * F[i-1, j  ] \
                     - 3. * U[i,   j-1] * F[i,   j-1] \
                     - 3. * U[i,   j-1] * F[i,   j  ] \
                     + 1. * U[i,   j+1] * F[i,   j+1] \
                     + 1. * U[i,   j+1] * F[i,   j+2] \
                     + 1. * U[i,   j  ] * F[i-1, j+1] \
                     + 1. * U[i,   j  ] * F[i-1, j  ] \
                     + 3. * U[i,   j  ] * F[i,   j+1] \
                     + 3. * U[i,   j  ] * F[i,   j  ] \
                 ) * self.hy_inv / 32.
         
        return result
    
    
    
    
    @cython.boundscheck(False)
    cdef double gradx(self, double[:,:] x,
                                  int i, int j):
        '''
        MHD Derivative: grad_x
        '''
        
        cdef double result
        
        result = ( \
                     + 1. * ( x[i+1, j-1] - x[i-1, j-1] ) \
                     + 2. * ( x[i+1, j  ] - x[i-1, j  ] ) \
                     + 1. * ( x[i+1, j+1] - x[i-1, j+1] ) \
                 ) * 0.25 * self.hx_inv / 2.
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef double grady(self, double[:,:] x,
                                  int i, int j):
        '''
        MHD Derivative: grad_y
        '''
        
        cdef double result
        
        result = ( \
                     + 1. * ( x[i-1, j+1] - x[i-1, j-1] ) \
                     + 2. * ( x[i,   j+1] - x[i,   j-1] ) \
                     + 1. * ( x[i+1, j+1] - x[i+1, j-1] ) \
                 ) * 0.25 * self.hy_inv / 2.

        return result
    
    
    
    @cython.boundscheck(False)
    cdef double gradx_sg(self, double[:,:] x,
                                     int i, int j):
        '''
        MHD Derivative: grad_x (staggered grid)
        '''
        
        cdef double result
        
        result = ( \
                     + 1. * ( x[i, j-1] - x[i-1, j-1] ) \
                     + 2. * ( x[i, j  ] - x[i-1, j  ] ) \
                     + 1. * ( x[i, j+1] - x[i-1, j+1] ) \
                 ) * 0.25 * self.hx_inv
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef double grady_sg(self, double[:,:] x,
                                     int i, int j):
        '''
        MHD Derivative: grad_y (staggered grid)
        '''
        
        cdef double result
        
        result = ( \
                     + 1. * ( x[i-1, j] - x[i-1, j-1] ) \
                     + 2. * ( x[i,   j] - x[i,   j-1] ) \
                     + 1. * ( x[i+1, j] - x[i+1, j-1] ) \
                 ) * 0.25 * self.hy_inv

        return result
    
    
    @cython.boundscheck(False)
    cdef double gradx_fv(self, double[:,:] x,
                                     int i, int j):
        '''
        MHD Derivative: grad_x (finite volume discretisation)
        '''
        
        cdef double result
        
        result = ( \
                     + 3. * ( x[i+1, j  ] - x[i,   j  ] ) \
                     + 1. * ( x[i+1, j-1] - x[i,   j-1] ) \
                     + 1. * ( x[i+1, j+1] - x[i,   j+1] ) \
                     + 1. * ( x[i+2, j  ] - x[i-1, j  ] ) \
                 ) * self.hx_inv / 8.
 
        return result
    
    
    @cython.boundscheck(False)
    cdef double grady_fv(self, double[:,:] x,
                                     int i, int j):
        '''
        MHD Derivative: grad_y (finite volume discretisation)
        '''
        
        cdef double result
        
        result = ( \
                     + 3. * ( x[i,   j+1] - x[i,   j  ] ) \
                     + 1. * ( x[i-1, j+1] - x[i-1, j  ] ) \
                     + 1. * ( x[i+1, j+1] - x[i+1, j  ] ) \
                     + 1. * ( x[i,   j+2] - x[i,   j-1] ) \
                 ) * self.hy_inv / 8.

        return result

    
    @cython.boundscheck(False)
    cdef double gradx_simple(self, double[:,:] x,
                                         int i, int j):
        '''
        MHD Derivative: grad_x (staggered grid)
        '''
        
        cdef double result
        
        result = ( x[i+1, j  ] - x[i, j  ] ) * self.hx_inv
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef double grady_simple(self, double[:,:] x,
                                         int i, int j):
        '''
        MHD Derivative: grad_y (staggered grid)
        '''
        
        cdef double result
        
        result = ( x[i,   j+1] - x[i,   j] ) * self.hy_inv

        return result
    
    
    @cython.boundscheck(False)
    cdef double divx(self, double[:,:] x,
                                  int i, int j):
        '''
        MHD Derivative: div_x
        '''
        
        cdef double result
        
        result = 0.5 * ( x[i+1, j] - x[i-1, j] ) * self.hx_inv
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef double divy(self, double[:,:] x,
                                  int i, int j):
        '''
        MHD Derivative: div_y
        '''
        
        cdef double result
        
        result = 0.5 * ( x[i, j+1] - x[i, j-1] ) * self.hy_inv

        return result
    
    
    
    @cython.boundscheck(False)
    cdef double divx_sg(self, double[:,:] x,
                                    int i, int j):
        '''
        MHD Derivative: div_x (staggered grid)
        '''
        
        return ( x[i, j] - x[i-1, j] ) * self.hx_inv
    
    
    
    @cython.boundscheck(False)
    cdef double divy_sg(self, double[:,:] x,
                                    int i, int j):
        '''
        MHD Derivative: div_y (staggered grid)
        '''
        
        return ( x[i, j] - x[i, j-1] ) * self.hy_inv
    
    
    @cython.boundscheck(False)
    cdef double laplace(self, double[:,:] x,
                                    int i, int j):
        '''
        MHD Derivative: Laplace operator (averaged)
        '''
        
        cdef double result
        
        result = 0.25 * ( \
                 ( \
                   + 1. * x[i-1, j-1] \
                   - 2. * x[i,   j-1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i-1, j  ] \
                   - 4. * x[i,   j  ] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i-1, j+1] \
                   - 2. * x[i,   j+1] \
                   + 1. * x[i+1, j+1] \
                 ) * self.hx_inv2 \
               + ( \
                   + 1. * x[i-1, j-1] \
                   - 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   - 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   - 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) * self.hy_inv2 \
               )
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef double laplace_fd(self, double[:,:] x,
                                       int i, int j):
        '''
        MHD Derivative: Laplace operator (averaged)
        '''
        
        cdef double result
        
        result = ( \
                   + 1. * x[i-1, j  ] \
                   - 2. * x[i,   j  ] \
                   + 1. * x[i+1, j  ] \
                 ) * self.hx_inv2 \
               + \
                 ( \
                   + 1. * x[i,   j-1] \
                   - 2. * x[i,   j  ] \
                   + 1. * x[i,   j+1] \
                 ) * self.hy_inv2
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef double dt(self, double[:,:] x,
                               int i, int j):
        '''
        Time Derivative
        '''
        
        cdef double result
        
        result = ( \
                   + 1. * x[i-1, j-1] \
                   + 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   + 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) * self.ht_inv / 16.
        
        return result


    @cython.boundscheck(False)
    cdef double dt_diag(self, double[:,:] x,
                                    int i, int j):
        '''
        Time Derivative
        '''
        
        cdef double result
        
        result = 0.25 * x[i, j] * self.ht_inv
        
        return result




    @cython.boundscheck(False)
    cdef double rot(self, double[:,:] Ux, double[:,:] Uy, int i, int j):

        cdef double result
        
        result = ( \
                   + ( Uy[i, j] - Uy[i-1, j  ] ) * self.hx_inv \
                   - ( Ux[i, j] - Ux[i,   j-1] ) * self.hy_inv \
                 ) 
        
        return result


    @cython.boundscheck(False)
    cdef double psix(self, double[:,:] Ux, double[:,:] Uy,
                           double[:,:] Vx, double[:,:] Vy,
                           int i, int j):

        cdef double result
        
        result = 0.25 * ( \
                   - ( Uy[i-1, j  ] + Uy[i, j  ] ) * self.rot(Vx, Vy, i, j  ) \
                   - ( Uy[i-1, j+1] + Uy[i, j+1] ) * self.rot(Vx, Vy, i, j+1) \
                 ) 
        
        return result


    @cython.boundscheck(False)
    cdef double psiy(self, double[:,:] Ux, double[:,:] Uy,
                           double[:,:] Vx, double[:,:] Vy,
                           int i, int j):


        cdef double result
        
        result = 0.25 * ( \
                   + ( Ux[i,   j-1] + Ux[i,   j] ) * self.rot(Vx, Vy, i,   j) \
                   + ( Ux[i+1, j-1] + Ux[i+1, j] ) * self.rot(Vx, Vy, i+1, j) \
                 )
        
        return result
    
    
    
    @cython.boundscheck(False)
    cdef double phix(self, double[:,:] Fx, double[:,:] Uy, int i, int j):

        cdef double result
        
        result = 0.25 * ( \
                   - ( Fx[i,   j  ] + Fx[i,   j+1] ) * ( Uy[i-1, j+1] + Uy[i,   j+1] ) \
                   + ( Fx[i,   j-1] + Fx[i,   j  ] ) * ( Uy[i-1, j  ] + Uy[i,   j  ] ) \
                 ) * self.hy_inv
        
        return result


    @cython.boundscheck(False)
    cdef double phiy(self, double[:,:] Fx, double[:,:] Uy, int i, int j):

        cdef double result
        
        result = 0.25 * ( \
                   + ( Fx[i+1, j-1] + Fx[i+1, j] ) * ( Uy[i,   j  ] + Uy[i+1, j] ) \
                   - ( Fx[i,   j-1] + Fx[i,   j] ) * ( Uy[i-1, j  ] + Uy[i,   j] ) \
                 ) * self.hx_inv
        
        return result



    @cython.boundscheck(False)
    cpdef double Bix(self, double[:,:] Bx, double[:,:] By, int i, int j, double de):

        cdef double result
        
        result = Bx[i,j] \
               + de*de * ( (By[i, j+1] - By[i-1, j+1])
                         - (By[i, j  ] - By[i-1, j  ]) ) * self.hx_inv * self.hy_inv \
               - de*de * ( Bx[i, j+1] - 2. * Bx[i,   j  ] + Bx[i, j-1] ) * self.hy_inv * self.hy_inv
        
        return result


    @cython.boundscheck(False)
    cpdef double Biy(self, double[:,:] Bx, double[:,:] By, int i, int j, double de):

        cdef double result
        
        result = By[i,j] \
               + de*de * ( (Bx[i+1, j] - Bx[i+1, j-1])
                         - (Bx[i,   j] - Bx[i,   j-1]) ) * self.hx_inv * self.hy_inv \
               - de*de * ( By[i+1, j] - 2. * By[i,   j  ] + By[i-1, j] ) * self.hx_inv * self.hx_inv
        
        return result
    
