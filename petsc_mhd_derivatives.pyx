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
        
        
        
#    @cython.boundscheck(False)
    cdef np.float64_t dx(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dx
        '''
        
        cdef np.float64_t result
        
#        result = ( \
#                     + 4. * B[i+1, j  ] * V[i+1, j  ] \
#                     - 4. * B[i-1, j  ] * V[i-1, j  ] \
#                     + 2. * B[i+1, j-1] * V[i+1, j-1] \
#                     - 2. * B[i-1, j-1] * V[i-1, j-1] \
#                     + 2. * B[i+1, j-1] * V[i+1, j  ] \
#                     - 2. * B[i-1, j-1] * V[i-1, j  ] \
#                     + 2. * B[i+1, j+1] * V[i+1, j+1] \
#                     - 2. * B[i-1, j+1] * V[i-1, j+1] \
#                     + 2. * B[i+1, j+1] * V[i+1, j  ] \
#                     - 2. * B[i-1, j+1] * V[i-1, j  ] \
#                     + 2. * B[i+1, j  ] * V[i+1, j-1] \
#                     - 2. * B[i-1, j  ] * V[i-1, j-1] \
#                     + 2. * B[i+1, j  ] * V[i+1, j+1] \
#                     - 2. * B[i-1, j  ] * V[i-1, j+1] \
#                     + 2. * B[i+1, j  ] * V[i,   j  ] \
#                     - 2. * B[i-1, j  ] * V[i,   j  ] \
#                     + 2. * B[i,   j  ] * V[i+1, j  ] \
#                     - 2. * B[i,   j  ] * V[i-1, j  ] \
#                     + 1. * B[i+1, j-1] * V[i,   j-1] \
#                     - 1. * B[i-1, j-1] * V[i,   j-1] \
#                     + 1. * B[i+1, j-1] * V[i,   j  ] \
#                     - 1. * B[i-1, j-1] * V[i,   j  ] \
#                     + 1. * B[i+1, j+1] * V[i,   j+1] \
#                     - 1. * B[i-1, j+1] * V[i,   j+1] \
#                     + 1. * B[i+1, j+1] * V[i,   j  ] \
#                     - 1. * B[i-1, j+1] * V[i,   j  ] \
#                     + 1. * B[i+1, j  ] * V[i,   j-1] \
#                     - 1. * B[i-1, j  ] * V[i,   j-1] \
#                     + 1. * B[i+1, j  ] * V[i,   j+1] \
#                     - 1. * B[i-1, j  ] * V[i,   j+1] \
#                     + 1. * B[i,   j-1] * V[i+1, j-1] \
#                     - 1. * B[i,   j-1] * V[i-1, j-1] \
#                     + 1. * B[i,   j-1] * V[i+1, j  ] \
#                     - 1. * B[i,   j-1] * V[i-1, j  ] \
#                     + 1. * B[i,   j+1] * V[i+1, j+1] \
#                     - 1. * B[i,   j+1] * V[i-1, j+1] \
#                     + 1. * B[i,   j+1] * V[i+1, j  ] \
#                     - 1. * B[i,   j+1] * V[i-1, j  ] \
#                     + 1. * B[i,   j  ] * V[i+1, j+1] \
#                     - 1. * B[i,   j  ] * V[i-1, j+1] \
#                     + 1. * B[i,   j  ] * V[i+1, j-1] \
#                     - 1. * B[i,   j  ] * V[i-1, j-1] \
#                 ) * self.hx_inv / 192.

        result = ( \
                     - 4 * V[i-1, j  ] * B[i-1, j  ]
                     + 4 * V[i+1, j  ] * B[i+1, j  ]
                     - 2 * V[i-1, j-1] * B[i-1, j-1]
                     - 2 * V[i-1, j-1] * B[i-1, j  ]
                     - 2 * V[i-1, j+1] * B[i-1, j+1]
                     - 2 * V[i-1, j+1] * B[i-1, j  ]
                     - 2 * V[i-1, j  ] * B[i-1, j-1]
                     - 2 * V[i-1, j  ] * B[i-1, j+1]
                     - 2 * V[i-1, j  ] * B[i,   j  ]
                     + 2 * V[i+1, j-1] * B[i+1, j-1]
                     + 2 * V[i+1, j-1] * B[i+1, j  ]
                     + 2 * V[i+1, j+1] * B[i+1, j+1]
                     + 2 * V[i+1, j+1] * B[i+1, j  ]
                     + 2 * V[i+1, j  ] * B[i+1, j-1]
                     + 2 * V[i+1, j  ] * B[i+1, j+1]
                     + 2 * V[i+1, j  ] * B[i,   j  ]
                     - 2 * V[i,   j  ] * B[i-1, j  ]
                     + 2 * V[i,   j  ] * B[i+1, j  ]
                     - 1 * V[i-1, j-1] * B[i,   j-1]
                     - 1 * V[i-1, j-1] * B[i,   j  ]
                     - 1 * V[i-1, j+1] * B[i,   j+1]
                     - 1 * V[i-1, j+1] * B[i,   j  ]
                     - 1 * V[i-1, j  ] * B[i,   j-1]
                     - 1 * V[i-1, j  ] * B[i,   j+1]
                     + 1 * V[i+1, j-1] * B[i,   j-1]
                     + 1 * V[i+1, j-1] * B[i,   j  ]
                     + 1 * V[i+1, j+1] * B[i,   j+1]
                     + 1 * V[i+1, j+1] * B[i,   j  ]
                     + 1 * V[i+1, j  ] * B[i,   j-1]
                     + 1 * V[i+1, j  ] * B[i,   j+1]
                     - 1 * V[i,   j-1] * B[i-1, j-1]
                     - 1 * V[i,   j-1] * B[i-1, j  ]
                     + 1 * V[i,   j-1] * B[i+1, j-1]
                     + 1 * V[i,   j-1] * B[i+1, j  ]
                     - 1 * V[i,   j+1] * B[i-1, j+1]
                     - 1 * V[i,   j+1] * B[i-1, j  ]
                     + 1 * V[i,   j+1] * B[i+1, j+1]
                     + 1 * V[i,   j+1] * B[i+1, j  ]
                     - 1 * V[i,   j  ] * B[i-1, j-1]
                     - 1 * V[i,   j  ] * B[i-1, j+1]
                     + 1 * V[i,   j  ] * B[i+1, j-1]
                     + 1 * V[i,   j  ] * B[i+1, j+1]
                  ) * self.hx_inv / 48.

        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dx1(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dx1
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 2. * B[i-1, j  ] * V[i-1, j  ] \
                     - 2. * B[i+1, j  ] * V[i+1, j  ] \
                     + 1. * B[i-1, j-1] * V[i-1, j-1] \
                     + 1. * B[i-1, j-1] * V[i-1, j  ] \
                     + 1. * B[i-1, j+1] * V[i-1, j+1] \
                     + 1. * B[i-1, j+1] * V[i-1, j  ] \
                     + 1. * B[i-1, j  ] * V[i-1, j-1] \
                     + 1. * B[i-1, j  ] * V[i-1, j+1] \
                     - 1. * B[i+1, j-1] * V[i+1, j-1] \
                     - 1. * B[i+1, j-1] * V[i+1, j  ] \
                     - 1. * B[i+1, j+1] * V[i+1, j+1] \
                     - 1. * B[i+1, j+1] * V[i+1, j  ] \
                     - 1. * B[i+1, j  ] * V[i+1, j-1] \
                     - 1. * B[i+1, j  ] * V[i+1, j+1] \
                 ) * self.hx_inv / 16.
 
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t fdudx(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative F d_x D
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     - 1 * F[i-1, j-1] * U[i-1, j-1] \
                     - 1 * F[i-1, j-1] * U[i-1, j  ] \
                     + 1 * F[i-1, j-1] * U[i,   j-1] \
                     + 1 * F[i-1, j-1] * U[i,   j  ] \
                     - 1 * F[i-1, j+1] * U[i-1, j+1] \
                     - 1 * F[i-1, j+1] * U[i-1, j  ] \
                     + 1 * F[i-1, j+1] * U[i,   j+1] \
                     + 1 * F[i-1, j+1] * U[i,   j  ] \
                     - 1 * F[i-1, j  ] * U[i-1, j-1] \
                     - 1 * F[i-1, j  ] * U[i-1, j+1] \
                     - 2 * F[i-1, j  ] * U[i-1, j  ] \
                     + 1 * F[i-1, j  ] * U[i,   j-1] \
                     + 1 * F[i-1, j  ] * U[i,   j+1] \
                     + 2 * F[i-1, j  ] * U[i,   j  ] \
                     + 1 * F[i+1, j-1] * U[i+1, j-1] \
                     + 1 * F[i+1, j-1] * U[i+1, j  ] \
                     - 1 * F[i+1, j-1] * U[i,   j-1] \
                     - 1 * F[i+1, j-1] * U[i,   j  ] \
                     + 1 * F[i+1, j+1] * U[i+1, j+1] \
                     + 1 * F[i+1, j+1] * U[i+1, j  ] \
                     - 1 * F[i+1, j+1] * U[i,   j+1] \
                     - 1 * F[i+1, j+1] * U[i,   j  ] \
                     + 1 * F[i+1, j  ] * U[i+1, j-1] \
                     + 1 * F[i+1, j  ] * U[i+1, j+1] \
                     + 2 * F[i+1, j  ] * U[i+1, j  ] \
                     - 1 * F[i+1, j  ] * U[i,   j-1] \
                     - 1 * F[i+1, j  ] * U[i,   j+1] \
                     - 2 * F[i+1, j  ] * U[i,   j  ] \
                     - 1 * F[i,   j-1] * U[i-1, j-1] \
                     - 1 * F[i,   j-1] * U[i-1, j  ] \
                     + 1 * F[i,   j-1] * U[i+1, j-1] \
                     + 1 * F[i,   j-1] * U[i+1, j  ] \
                     - 1 * F[i,   j+1] * U[i-1, j+1] \
                     - 1 * F[i,   j+1] * U[i-1, j  ] \
                     + 1 * F[i,   j+1] * U[i+1, j+1] \
                     + 1 * F[i,   j+1] * U[i+1, j  ] \
                     - 1 * F[i,   j  ] * U[i-1, j-1] \
                     - 1 * F[i,   j  ] * U[i-1, j+1] \
                     - 2 * F[i,   j  ] * U[i-1, j  ] \
                     + 1 * F[i,   j  ] * U[i+1, j-1] \
                     + 1 * F[i,   j  ] * U[i+1, j+1] \
                     + 2 * F[i,   j  ] * U[i+1, j  ] \
                 ) * self.hx_inv / 32.
         
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dx3(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dx3
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     - 2 * V[i-1, j  ] * B[i,   j  ]
                     + 2 * V[i+1, j  ] * B[i,   j  ]
                     - 2 * V[i,   j  ] * B[i-1, j  ]
                     + 2 * V[i,   j  ] * B[i+1, j  ]
                     - 1 * V[i-1, j-1] * B[i,   j-1]
                     - 1 * V[i-1, j-1] * B[i,   j  ]
                     - 1 * V[i-1, j+1] * B[i,   j+1]
                     - 1 * V[i-1, j+1] * B[i,   j  ]
                     - 1 * V[i-1, j  ] * B[i,   j-1]
                     - 1 * V[i-1, j  ] * B[i,   j+1]
                     + 1 * V[i+1, j-1] * B[i,   j-1]
                     + 1 * V[i+1, j-1] * B[i,   j  ]
                     + 1 * V[i+1, j+1] * B[i,   j+1]
                     + 1 * V[i+1, j+1] * B[i,   j  ]
                     + 1 * V[i+1, j  ] * B[i,   j-1]
                     + 1 * V[i+1, j  ] * B[i,   j+1]
                     - 1 * V[i,   j-1] * B[i-1, j-1]
                     - 1 * V[i,   j-1] * B[i-1, j  ]
                     + 1 * V[i,   j-1] * B[i+1, j-1]
                     + 1 * V[i,   j-1] * B[i+1, j  ]
                     - 1 * V[i,   j+1] * B[i-1, j+1]
                     - 1 * V[i,   j+1] * B[i-1, j  ]
                     + 1 * V[i,   j+1] * B[i+1, j+1]
                     + 1 * V[i,   j+1] * B[i+1, j  ]
                     - 1 * V[i,   j  ] * B[i-1, j-1]
                     - 1 * V[i,   j  ] * B[i-1, j+1]
                     + 1 * V[i,   j  ] * B[i+1, j-1]
                     + 1 * V[i,   j  ] * B[i+1, j+1]
                 ) * self.hx_inv / 16.
 
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dy(self, np.ndarray[np.float64_t, ndim=2] B,
                               np.ndarray[np.float64_t, ndim=2] V,
                               np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dy
        '''
        
        cdef np.float64_t result
        
#        result = ( \
#                     + 4. * B[i,   j+1] * V[i,   j+1] \
#                     - 4. * B[i,   j-1] * V[i,   j-1] \
#                     + 2. * B[i,   j+1] * V[i,   j  ] \
#                     - 2. * B[i,   j-1] * V[i,   j  ] \
#                     + 2. * B[i,   j  ] * V[i,   j+1] \
#                     - 2. * B[i,   j  ] * V[i,   j-1] \
#                     + 2. * B[i,   j+1] * V[i+1, j+1] \
#                     - 2. * B[i,   j-1] * V[i-1, j-1] \
#                     + 2. * B[i,   j+1] * V[i-1, j+1] \
#                     - 2. * B[i,   j-1] * V[i+1, j-1] \
#                     + 2. * B[i-1, j+1] * V[i-1, j+1] \
#                     - 2. * B[i-1, j-1] * V[i-1, j-1] \
#                     + 2. * B[i-1, j+1] * V[i,   j+1] \
#                     - 2. * B[i-1, j-1] * V[i,   j-1] \
#                     + 2. * B[i+1, j+1] * V[i+1, j+1] \
#                     - 2. * B[i+1, j-1] * V[i+1, j-1] \
#                     + 2. * B[i+1, j+1] * V[i,   j+1] \
#                     - 2. * B[i+1, j-1] * V[i,   j-1] \
#                     + 1. * B[i-1, j+1] * V[i-1, j  ] \
#                     - 1. * B[i-1, j-1] * V[i-1, j  ] \
#                     + 1. * B[i-1, j+1] * V[i,   j  ] \
#                     - 1. * B[i-1, j-1] * V[i,   j  ] \
#                     + 1. * B[i-1, j  ] * V[i-1, j+1] \
#                     - 1. * B[i-1, j  ] * V[i-1, j-1] \
#                     + 1. * B[i-1, j  ] * V[i,   j+1] \
#                     - 1. * B[i-1, j  ] * V[i,   j-1] \
#                     + 1. * B[i+1, j+1] * V[i+1, j  ] \
#                     - 1. * B[i+1, j-1] * V[i+1, j  ] \
#                     + 1. * B[i+1, j+1] * V[i,   j  ] \
#                     - 1. * B[i+1, j-1] * V[i,   j  ] \
#                     + 1. * B[i+1, j  ] * V[i+1, j+1] \
#                     - 1. * B[i+1, j  ] * V[i+1, j-1] \
#                     + 1. * B[i+1, j  ] * V[i,   j+1] \
#                     - 1. * B[i+1, j  ] * V[i,   j-1] \
#                     + 1. * B[i,   j+1] * V[i+1, j  ] \
#                     - 1. * B[i,   j-1] * V[i+1, j  ] \
#                     + 1. * B[i,   j+1] * V[i-1, j  ] \
#                     - 1. * B[i,   j-1] * V[i-1, j  ] \
#                     + 1. * B[i,   j  ] * V[i-1, j+1] \
#                     - 1. * B[i,   j  ] * V[i-1, j-1] \
#                     + 1. * B[i,   j  ] * V[i+1, j+1] \
#                     - 1. * B[i,   j  ] * V[i+1, j-1] \
#                 ) * self.hy_inv / 192.
                 
        result = ( \
                     - 4 * V[i,   j-1] * B[i,   j-1]
                     + 4 * V[i,   j+1] * B[i,   j+1]
                     - 2 * V[i-1, j-1] * B[i-1, j-1]
                     - 2 * V[i-1, j-1] * B[i,   j-1]
                     + 2 * V[i-1, j+1] * B[i-1, j+1]
                     + 2 * V[i-1, j+1] * B[i,   j+1]
                     - 2 * V[i+1, j-1] * B[i+1, j-1]
                     - 2 * V[i+1, j-1] * B[i,   j-1]
                     + 2 * V[i+1, j+1] * B[i+1, j+1]
                     + 2 * V[i+1, j+1] * B[i,   j+1]
                     - 2 * V[i,   j-1] * B[i-1, j-1]
                     - 2 * V[i,   j-1] * B[i+1, j-1]
                     - 2 * V[i,   j-1] * B[i,   j  ]
                     + 2 * V[i,   j+1] * B[i-1, j+1]
                     + 2 * V[i,   j+1] * B[i+1, j+1]
                     + 2 * V[i,   j+1] * B[i,   j  ]
                     - 2 * V[i,   j  ] * B[i,   j-1]
                     + 2 * V[i,   j  ] * B[i,   j+1]
                     - 1 * V[i-1, j-1] * B[i-1, j  ]
                     - 1 * V[i-1, j-1] * B[i,   j  ]
                     + 1 * V[i-1, j+1] * B[i-1, j  ]
                     + 1 * V[i-1, j+1] * B[i,   j  ]
                     - 1 * V[i-1, j  ] * B[i-1, j-1]
                     + 1 * V[i-1, j  ] * B[i-1, j+1]
                     - 1 * V[i-1, j  ] * B[i,   j-1]
                     + 1 * V[i-1, j  ] * B[i,   j+1]
                     - 1 * V[i+1, j-1] * B[i+1, j  ]
                     - 1 * V[i+1, j-1] * B[i,   j  ]
                     + 1 * V[i+1, j+1] * B[i+1, j  ]
                     + 1 * V[i+1, j+1] * B[i,   j  ]
                     - 1 * V[i+1, j  ] * B[i+1, j-1]
                     + 1 * V[i+1, j  ] * B[i+1, j+1]
                     - 1 * V[i+1, j  ] * B[i,   j-1]
                     + 1 * V[i+1, j  ] * B[i,   j+1]
                     - 1 * V[i,   j-1] * B[i-1, j  ]
                     - 1 * V[i,   j-1] * B[i+1, j  ]
                     + 1 * V[i,   j+1] * B[i-1, j  ]
                     + 1 * V[i,   j+1] * B[i+1, j  ]
                     - 1 * V[i,   j  ] * B[i-1, j-1]
                     + 1 * V[i,   j  ] * B[i-1, j+1]
                     - 1 * V[i,   j  ] * B[i+1, j-1]
                     + 1 * V[i,   j  ] * B[i+1, j+1]                 
                 ) * self.hy_inv / 48.

        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dy1(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dy1
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 2. * B[i,   j-1] * V[i,   j-1] \
                     - 2. * B[i,   j+1] * V[i,   j+1] \
                     + 1. * B[i-1, j-1] * V[i-1, j-1] \
                     + 1. * B[i-1, j-1] * V[i,   j-1] \
                     - 1. * B[i-1, j+1] * V[i-1, j+1] \
                     - 1. * B[i-1, j+1] * V[i,   j+1] \
                     + 1. * B[i+1, j-1] * V[i+1, j-1] \
                     + 1. * B[i+1, j-1] * V[i,   j-1] \
                     - 1. * B[i+1, j+1] * V[i+1, j+1] \
                     - 1. * B[i+1, j+1] * V[i,   j+1] \
                     + 1. * B[i,   j-1] * V[i-1, j-1] \
                     + 1. * B[i,   j-1] * V[i+1, j-1] \
                     - 1. * B[i,   j+1] * V[i-1, j+1] \
                     - 1. * B[i,   j+1] * V[i+1, j+1] \
                 ) * self.hy_inv / 16.
                 
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t fdudy(self, np.ndarray[np.float64_t, ndim=2] F,
                                  np.ndarray[np.float64_t, ndim=2] U,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: full single derivative F d_y D
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     - 1 * F[i-1, j-1] * U[i-1, j-1] \
                     + 1 * F[i-1, j-1] * U[i-1, j  ] \
                     - 1 * F[i-1, j-1] * U[i,   j-1] \
                     + 1 * F[i-1, j-1] * U[i,   j  ] \
                     + 1 * F[i-1, j+1] * U[i-1, j+1] \
                     - 1 * F[i-1, j+1] * U[i-1, j  ] \
                     + 1 * F[i-1, j+1] * U[i,   j+1] \
                     - 1 * F[i-1, j+1] * U[i,   j  ] \
                     - 1 * F[i-1, j  ] * U[i-1, j-1] \
                     + 1 * F[i-1, j  ] * U[i-1, j+1] \
                     - 1 * F[i-1, j  ] * U[i,   j-1] \
                     + 1 * F[i-1, j  ] * U[i,   j+1] \
                     - 1 * F[i+1, j-1] * U[i+1, j-1] \
                     + 1 * F[i+1, j-1] * U[i+1, j  ] \
                     - 1 * F[i+1, j-1] * U[i,   j-1] \
                     + 1 * F[i+1, j-1] * U[i,   j  ] \
                     + 1 * F[i+1, j+1] * U[i+1, j+1] \
                     - 1 * F[i+1, j+1] * U[i+1, j  ] \
                     + 1 * F[i+1, j+1] * U[i,   j+1] \
                     - 1 * F[i+1, j+1] * U[i,   j  ] \
                     - 1 * F[i+1, j  ] * U[i+1, j-1] \
                     + 1 * F[i+1, j  ] * U[i+1, j+1] \
                     - 1 * F[i+1, j  ] * U[i,   j-1] \
                     + 1 * F[i+1, j  ] * U[i,   j+1] \
                     - 1 * F[i,   j-1] * U[i-1, j-1] \
                     + 1 * F[i,   j-1] * U[i-1, j  ] \
                     - 1 * F[i,   j-1] * U[i+1, j-1] \
                     + 1 * F[i,   j-1] * U[i+1, j  ] \
                     - 2 * F[i,   j-1] * U[i,   j-1] \
                     + 2 * F[i,   j-1] * U[i,   j  ] \
                     + 1 * F[i,   j+1] * U[i-1, j+1] \
                     - 1 * F[i,   j+1] * U[i-1, j  ] \
                     + 1 * F[i,   j+1] * U[i+1, j+1] \
                     - 1 * F[i,   j+1] * U[i+1, j  ] \
                     + 2 * F[i,   j+1] * U[i,   j+1] \
                     - 2 * F[i,   j+1] * U[i,   j  ] \
                     - 1 * F[i,   j  ] * U[i-1, j-1] \
                     + 1 * F[i,   j  ] * U[i-1, j+1] \
                     - 1 * F[i,   j  ] * U[i+1, j-1] \
                     + 1 * F[i,   j  ] * U[i+1, j+1] \
                     - 2 * F[i,   j  ] * U[i,   j-1] \
                     + 2 * F[i,   j  ] * U[i,   j+1] \
                 ) * self.hy_inv / 32.
        
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dy3(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dy3
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     - 2 * V[i,   j-1] * B[i,   j  ]
                     + 2 * V[i,   j+1] * B[i,   j  ]
                     - 2 * V[i,   j  ] * B[i,   j-1]
                     + 2 * V[i,   j  ] * B[i,   j+1]
                     - 1 * V[i-1, j-1] * B[i-1, j  ]
                     - 1 * V[i-1, j-1] * B[i,   j  ]
                     + 1 * V[i-1, j+1] * B[i-1, j  ]
                     + 1 * V[i-1, j+1] * B[i,   j  ]
                     - 1 * V[i-1, j  ] * B[i-1, j-1]
                     + 1 * V[i-1, j  ] * B[i-1, j+1]
                     - 1 * V[i-1, j  ] * B[i,   j-1]
                     + 1 * V[i-1, j  ] * B[i,   j+1]
                     - 1 * V[i+1, j-1] * B[i+1, j  ]
                     - 1 * V[i+1, j-1] * B[i,   j  ]
                     + 1 * V[i+1, j+1] * B[i+1, j  ]
                     + 1 * V[i+1, j+1] * B[i,   j  ]
                     - 1 * V[i+1, j  ] * B[i+1, j-1]
                     + 1 * V[i+1, j  ] * B[i+1, j+1]
                     - 1 * V[i+1, j  ] * B[i,   j-1]
                     + 1 * V[i+1, j  ] * B[i,   j+1]
                     - 1 * V[i,   j-1] * B[i-1, j  ]
                     - 1 * V[i,   j-1] * B[i+1, j  ]
                     + 1 * V[i,   j+1] * B[i-1, j  ]
                     + 1 * V[i,   j+1] * B[i+1, j  ]
                     - 1 * V[i,   j  ] * B[i-1, j-1]
                     + 1 * V[i,   j  ] * B[i-1, j+1]
                     - 1 * V[i,   j  ] * B[i+1, j-1]
                     + 1 * V[i,   j  ] * B[i+1, j+1]
                 ) * self.hy_inv / 16.
                 
        return result
    
    
#    @cython.boundscheck(False)
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
                 ) * self.hx_inv / 4.
 
        return result
    
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t grady(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_y
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * ( x[i+1, j+1] - x[i+1, j-1] ) \
                     + 2. * ( x[i,   j+1] - x[i,   j-1] ) \
                     + 1. * ( x[i-1, j+1] - x[i-1, j-1] ) \
                 ) * self.hy_inv / 4.

        return result
    
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t laplace(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: Laplace operator (averaged)
        '''
        
        cdef np.float64_t result
        
#        result = ( \
#                   + 1. * x[i-1, j  ] \
#                   - 2. * x[i,   j  ] \
#                   + 1. * x[i+1, j  ] \
#                 ) / self.hx**2 \
#               + ( \
#                   + 1. * x[i,   j-1] \
#                   - 2. * x[i,   j  ] \
#                   + 1. * x[i,   j+1] \
#                 ) / self.hy**2
        
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
    
    
    
#    @cython.boundscheck(False)
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
        
#        result = x[i,j] * self.ht_inv
        
        return result


#    @cython.boundscheck(False)
    cdef np.float64_t dt_diag(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = 0.25 * x[i, j] * self.ht_inv
        
        return result


#    @cython.boundscheck(False)
    cdef np.float64_t dt_diag_inv(self, np.ndarray[np.float64_t, ndim=2] x,
                                        np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = 4.0  * self.ht / x[i, j]
        
        return result


