'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport DA, Vec


cdef class PETSc_MHD_Derivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    
    def __cinit__(self, DA da,
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
        
        
        # distributed array
        self.da = da
        
    
    
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
                 ) * self.hx_inv / 64.
 
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dx2(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dx2
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 2. * B[i-1, j  ] * V[i-1, j  ] \
                     - 2. * B[i-1, j  ] * V[i,   j  ] \
                     - 2. * B[i+1, j  ] * V[i+1, j  ] \
                     + 2. * B[i+1, j  ] * V[i,   j  ] \
                     + 2. * B[i,   j  ] * V[i-1, j  ] \
                     - 2. * B[i,   j  ] * V[i+1, j  ] \
                     + 1. * B[i-1, j-1] * V[i-1, j-1] \
                     + 1. * B[i-1, j-1] * V[i-1, j  ] \
                     - 1. * B[i-1, j-1] * V[i,   j-1] \
                     - 1. * B[i-1, j-1] * V[i,   j  ] \
                     + 1. * B[i-1, j+1] * V[i-1, j+1] \
                     + 1. * B[i-1, j+1] * V[i-1, j  ] \
                     - 1. * B[i-1, j+1] * V[i,   j+1] \
                     - 1. * B[i-1, j+1] * V[i,   j  ] \
                     + 1. * B[i-1, j  ] * V[i-1, j-1] \
                     + 1. * B[i-1, j  ] * V[i-1, j+1] \
                     - 1. * B[i-1, j  ] * V[i,   j-1] \
                     - 1. * B[i-1, j  ] * V[i,   j+1] \
                     - 1. * B[i+1, j-1] * V[i+1, j-1] \
                     - 1. * B[i+1, j-1] * V[i+1, j  ] \
                     + 1. * B[i+1, j-1] * V[i,   j-1] \
                     + 1. * B[i+1, j-1] * V[i,   j  ] \
                     - 1. * B[i+1, j+1] * V[i+1, j+1] \
                     - 1. * B[i+1, j+1] * V[i+1, j  ] \
                     + 1. * B[i+1, j+1] * V[i,   j+1] \
                     + 1. * B[i+1, j+1] * V[i,   j  ] \
                     - 1. * B[i+1, j  ] * V[i+1, j-1] \
                     - 1. * B[i+1, j  ] * V[i+1, j+1] \
                     + 1. * B[i+1, j  ] * V[i,   j-1] \
                     + 1. * B[i+1, j  ] * V[i,   j+1] \
                     + 1. * B[i,   j-1] * V[i-1, j-1] \
                     + 1. * B[i,   j-1] * V[i-1, j  ] \
                     - 1. * B[i,   j-1] * V[i+1, j-1] \
                     - 1. * B[i,   j-1] * V[i+1, j  ] \
                     + 1. * B[i,   j+1] * V[i-1, j+1] \
                     + 1. * B[i,   j+1] * V[i-1, j  ] \
                     - 1. * B[i,   j+1] * V[i+1, j+1] \
                     - 1. * B[i,   j+1] * V[i+1, j  ] \
                     + 1. * B[i,   j  ] * V[i-1, j-1] \
                     + 1. * B[i,   j  ] * V[i-1, j+1] \
                     - 1. * B[i,   j  ] * V[i+1, j-1] \
                     - 1. * B[i,   j  ] * V[i+1, j+1] \
                 ) * self.hx_inv / 128.
         
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
                     + 2. * B[i+1, j  ] * V[i+1, j  ] \
                     - 2. * B[i-1, j  ] * V[i-1, j  ] \
                     + 1. * B[i+1, j-1] * V[i+1, j-1] \
                     - 1. * B[i-1, j-1] * V[i-1, j-1] \
                     + 1. * B[i+1, j-1] * V[i+1, j  ] \
                     - 1. * B[i-1, j-1] * V[i-1, j  ] \
                     + 1. * B[i+1, j+1] * V[i+1, j+1] \
                     - 1. * B[i-1, j+1] * V[i-1, j+1] \
                     + 1. * B[i+1, j+1] * V[i+1, j  ] \
                     - 1. * B[i-1, j+1] * V[i-1, j  ] \
                 ) * self.hx_inv / 64.
        
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dx4(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dx4
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + B[i+1, j-1] * V[i+1, j  ] \
                     - B[i-1, j-1] * V[i-1, j  ] \
                     + B[i+1, j+1] * V[i+1, j  ] \
                     - B[i-1, j+1] * V[i-1, j  ] \
                 ) * self.hx_inv / 64.
        
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
                 ) * self.hy_inv / 64.
                 
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dy2(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dy2
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 2. * B[i,   j-1] * V[i,   j-1] \
                     + 2. * B[i,   j-1] * V[i,   j  ] \
                     - 2. * B[i,   j+1] * V[i,   j+1] \
                     - 2. * B[i,   j+1] * V[i,   j  ] \
                     - 2. * B[i,   j  ] * V[i,   j-1] \
                     + 2. * B[i,   j  ] * V[i,   j+1] \
                     + 1. * B[i-1, j-1] * V[i-1, j-1] \
                     + 1. * B[i-1, j-1] * V[i-1, j  ] \
                     + 1. * B[i-1, j-1] * V[i,   j-1] \
                     + 1. * B[i-1, j-1] * V[i,   j  ] \
                     - 1. * B[i-1, j+1] * V[i-1, j+1] \
                     - 1. * B[i-1, j+1] * V[i-1, j  ] \
                     - 1. * B[i-1, j+1] * V[i,   j+1] \
                     - 1. * B[i-1, j+1] * V[i,   j  ] \
                     - 1. * B[i-1, j  ] * V[i-1, j-1] \
                     + 1. * B[i-1, j  ] * V[i-1, j+1] \
                     - 1. * B[i-1, j  ] * V[i,   j-1] \
                     + 1. * B[i-1, j  ] * V[i,   j+1] \
                     + 1. * B[i+1, j-1] * V[i+1, j-1] \
                     + 1. * B[i+1, j-1] * V[i+1, j  ] \
                     + 1. * B[i+1, j-1] * V[i,   j-1] \
                     + 1. * B[i+1, j-1] * V[i,   j  ] \
                     - 1. * B[i+1, j+1] * V[i+1, j+1] \
                     - 1. * B[i+1, j+1] * V[i+1, j  ] \
                     - 1. * B[i+1, j+1] * V[i,   j+1] \
                     - 1. * B[i+1, j+1] * V[i,   j  ] \
                     - 1. * B[i+1, j  ] * V[i+1, j-1] \
                     + 1. * B[i+1, j  ] * V[i+1, j+1] \
                     - 1. * B[i+1, j  ] * V[i,   j-1] \
                     + 1. * B[i+1, j  ] * V[i,   j+1] \
                     + 1. * B[i,   j-1] * V[i-1, j-1] \
                     + 1. * B[i,   j-1] * V[i-1, j  ] \
                     + 1. * B[i,   j-1] * V[i+1, j-1] \
                     + 1. * B[i,   j-1] * V[i+1, j  ] \
                     - 1. * B[i,   j+1] * V[i-1, j+1] \
                     - 1. * B[i,   j+1] * V[i-1, j  ] \
                     - 1. * B[i,   j+1] * V[i+1, j+1] \
                     - 1. * B[i,   j+1] * V[i+1, j  ] \
                     - 1. * B[i,   j  ] * V[i-1, j-1] \
                     + 1. * B[i,   j  ] * V[i-1, j+1] \
                     - 1. * B[i,   j  ] * V[i+1, j-1] \
                     + 1. * B[i,   j  ] * V[i+1, j+1] \
                 ) * self.hy_inv / 128.
        
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
                     + 2. * B[i,   j+1] * V[i,   j+1] \
                     - 2. * B[i,   j-1] * V[i,   j-1] \
                     - 1. * B[i-1, j-1] * V[i-1, j-1] \
                     - 1. * B[i-1, j-1] * V[i,   j-1] \
                     + 1. * B[i-1, j+1] * V[i-1, j+1] \
                     + 1. * B[i-1, j+1] * V[i,   j+1] \
                     - 1. * B[i+1, j-1] * V[i+1, j-1] \
                     - 1. * B[i+1, j-1] * V[i,   j-1] \
                     + 1. * B[i+1, j+1] * V[i+1, j+1] \
                     + 1. * B[i+1, j+1] * V[i,   j+1] \
                 ) * self.hy_inv / 64.
        
        return result
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t dy4(self, np.ndarray[np.float64_t, ndim=2] B,
                                np.ndarray[np.float64_t, ndim=2] V,
                                np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: dy4
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + B[i-1, j+1] * V[i,   j+1] \
                     - B[i-1, j-1] * V[i,   j-1] \
                     + B[i+1, j+1] * V[i,   j+1] \
                     - B[i+1, j-1] * V[i,   j-1] \
                 ) * self.hy_inv / 64.

        return result
    
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t gradx(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_x
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 2. * ( x[i+1, j  ] - x[i-1, j  ] ) \
                     + 1. * ( x[i+1, j-1] - x[i-1, j-1] ) \
                     + 1. * ( x[i+1, j+1] - x[i-1, j+1] ) \
                 ) * self.hx_inv / 32.
 
        return result
    
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t grady(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: grad_y
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 2. * ( x[i,   j+1] - x[i,   j-1] ) \
                     + 1. * ( x[i+1, j+1] - x[i+1, j-1] ) \
                     + 1. * ( x[i-1, j+1] - x[i-1, j-1] ) \
                 ) * self.hy_inv / 32.

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
                 ) / (32. * self.ht)
        
        return result
    
    
    
#    @cython.boundscheck(False)
    cdef timestep(self, np.ndarray[np.float64_t, ndim=3] tx,
                        np.ndarray[np.float64_t, ndim=3] ty):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        
        for j in np.arange(ys, ye):
            jx = iy-ys+1
            jy = iy-ys
            
            for i in np.arange(xs, xe):
                ix = ix-xs+1
                iy = ix-xs
                
                ty[i, j, :] = 0.0
    
    
