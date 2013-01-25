'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np


cdef class PETSc_MHD_Derivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    
    def __cinit__(self,
                  np.uint64_t  nx, np.uint64_t  ny,
                  np.float64_t ht, np.float64_t hx, np.float64_t hy):
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
    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dx
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t fdudx(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative F d_x D
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t fx_dx_ux(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative F_x d_x U_x
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t fx_dx_uy(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative F_x d_x U_y
        '''
        
        cdef np.float64_t result
        
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
    
        
    
    cdef np.float64_t dx_fx_uy(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative d_x F_x U_y
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dy
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t fdudy(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative F d_y D
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t fy_dy_ux(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative F_y d_y U_x
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t fy_dy_uy(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative F_y d_y U_y
        '''
        
        cdef np.float64_t result
        
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
    
        
    
    cdef np.float64_t dy_fx_uy(self, np.ndarray[np.float64_t, ndim=2] F,
                                     np.ndarray[np.float64_t, ndim=2] U,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative d_y F_x U_y
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t gradx(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_x
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * ( x[i+1, j-1] - x[i-1, j-1] ) \
                     + 2. * ( x[i+1, j  ] - x[i-1, j  ] ) \
                     + 1. * ( x[i+1, j+1] - x[i-1, j+1] ) \
                 ) * 0.25 * self.hx_inv / 2.
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t grady(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_y
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * ( x[i-1, j+1] - x[i-1, j-1] ) \
                     + 2. * ( x[i,   j+1] - x[i,   j-1] ) \
                     + 1. * ( x[i+1, j+1] - x[i+1, j-1] ) \
                 ) * 0.25 * self.hy_inv / 2.

        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t gradx_sg(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_x (staggered grid)
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * ( x[i, j-1] - x[i-1, j-1] ) \
                     + 2. * ( x[i, j  ] - x[i-1, j  ] ) \
                     + 1. * ( x[i, j+1] - x[i-1, j+1] ) \
                 ) * 0.25 * self.hx_inv
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t grady_sg(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_y (staggered grid)
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * ( x[i-1, j] - x[i-1, j-1] ) \
                     + 2. * ( x[i,   j] - x[i,   j-1] ) \
                     + 1. * ( x[i+1, j] - x[i+1, j-1] ) \
                 ) * 0.25 * self.hy_inv

        return result
    
    
    @cython.boundscheck(False)
    cdef np.float64_t gradx_fv(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_x (finite volume discretisation)
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 3. * ( x[i+1, j  ] - x[i,   j  ] ) \
                     + 1. * ( x[i+1, j-1] - x[i,   j-1] ) \
                     + 1. * ( x[i+1, j+1] - x[i,   j+1] ) \
                     + 1. * ( x[i+2, j  ] - x[i-1, j  ] ) \
                 ) * self.hx_inv / 8.
 
        return result
    
    
    @cython.boundscheck(False)
    cdef np.float64_t grady_fv(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_y (finite volume discretisation)
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 3. * ( x[i,   j+1] - x[i,   j  ] ) \
                     + 1. * ( x[i-1, j+1] - x[i-1, j  ] ) \
                     + 1. * ( x[i+1, j+1] - x[i+1, j  ] ) \
                     + 1. * ( x[i,   j+2] - x[i,   j-1] ) \
                 ) * self.hy_inv / 8.

        return result

    
    @cython.boundscheck(False)
    cdef np.float64_t gradx_simple(self, np.ndarray[np.float64_t, ndim=2] x,
                                         np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_x (staggered grid)
        '''
        
        cdef np.float64_t result
        
        result = ( x[i+1, j  ] - x[i, j  ] ) * self.hx_inv
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t grady_simple(self, np.ndarray[np.float64_t, ndim=2] x,
                                         np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_y (staggered grid)
        '''
        
        cdef np.float64_t result
        
        result = ( x[i,   j+1] - x[i,   j] ) * self.hy_inv

        return result
    
    
    @cython.boundscheck(False)
    cdef np.float64_t divx(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: div_x
        '''
        
        cdef np.float64_t result
        
        result = 0.5 * ( x[i+1, j] - x[i-1, j] ) * self.hx_inv
 
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t divy(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: div_y
        '''
        
        cdef np.float64_t result
        
        result = 0.5 * ( x[i, j+1] - x[i, j-1] ) * self.hy_inv

        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t divx_sg(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: div_x (staggered grid)
        '''
        
        return ( x[i, j] - x[i-1, j] ) * self.hx_inv
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t divy_sg(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: div_y (staggered grid)
        '''
        
        return ( x[i, j] - x[i, j-1] ) * self.hy_inv
    
    
    @cython.boundscheck(False)
    cdef np.float64_t laplace(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: Laplace operator (averaged)
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t laplace_fd(self, np.ndarray[np.float64_t, ndim=2] x,
                                       np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: Laplace operator (averaged)
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t dt(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
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
    cdef np.float64_t dt_diag(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * x[i, j] * self.ht_inv
        
        return result




    @cython.boundscheck(False)
    cdef np.float64_t rot(self, np.ndarray[np.float64_t, ndim=2] Ux,
                                np.ndarray[np.float64_t, ndim=2] Uy,
                                np.uint64_t i, np.uint64_t j):

        cdef np.float64_t result
        
        result = ( \
                   + ( Uy[i, j] - Uy[i-1, j  ] ) * self.hx_inv \
                   - ( Ux[i, j] - Ux[i,   j-1] ) * self.hy_inv \
                 ) 
        
        return result


    @cython.boundscheck(False)
    cdef np.float64_t psix(self, np.ndarray[np.float64_t, ndim=2] Ux,
                                 np.ndarray[np.float64_t, ndim=2] Uy,
                                 np.uint64_t i, np.uint64_t j):

        cdef np.float64_t result
        
        result = 0.25 * ( \
                   - ( Uy[i-1, j  ] + Uy[i, j  ] ) * self.rot(Ux, Uy, i, j  ) \
                   - ( Uy[i-1, j+1] + Uy[i, j+1] ) * self.rot(Ux, Uy, i, j+1) \
                 ) 
        
        return result


    @cython.boundscheck(False)
    cdef np.float64_t psiy(self, np.ndarray[np.float64_t, ndim=2] Ux,
                                 np.ndarray[np.float64_t, ndim=2] Uy,
                                 np.uint64_t i, np.uint64_t j):


        cdef np.float64_t result
        
        result = 0.25 * ( \
                   + ( Ux[i,   j-1] + Ux[i,   j] ) * self.rot(Ux, Uy, i,   j) \
                   + ( Ux[i+1, j-1] + Ux[i+1, j] ) * self.rot(Ux, Uy, i+1, j) \
                 )
        
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t phix(self, np.ndarray[np.float64_t, ndim=2] Fx,
                                 np.ndarray[np.float64_t, ndim=2] Uy,
                                 np.uint64_t i, np.uint64_t j):

        cdef np.float64_t result
        
        result = 0.25 * ( \
                   - ( Fx[i,   j  ] + Fx[i,   j+1] ) * ( Uy[i-1, j+1] + Uy[i,   j+1] ) \
                   + ( Fx[i,   j-1] + Fx[i,   j  ] ) * ( Uy[i-1, j  ] + Uy[i,   j  ] ) \
                 ) * self.hy_inv
        
        return result


    @cython.boundscheck(False)
    cdef np.float64_t phiy(self, np.ndarray[np.float64_t, ndim=2] Fx,
                                 np.ndarray[np.float64_t, ndim=2] Uy,
                                 np.uint64_t i, np.uint64_t j):


        cdef np.float64_t result
        
        result = 0.25 * ( \
                   + ( Fx[i+1, j-1] + Fx[i+1, j] ) * ( Uy[i,   j  ] + Uy[i+1, j] ) \
                   - ( Fx[i,   j-1] + Fx[i,   j] ) * ( Uy[i-1, j  ] + Uy[i,   j] ) \
                 ) * self.hx_inv
        
        return result
