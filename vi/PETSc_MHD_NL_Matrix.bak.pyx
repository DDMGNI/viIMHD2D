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
    
    def __init__(self, DA da1, DA da4,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy,
                 np.float64_t omega):
        '''
        Constructor
        '''
        
        assert da1.getDim() == 2
        assert da4.getDim() == 2
        
        # distributed array
        self.da1 = da1
        self.da4 = da4
        
        # grid size
        self.nx = nx
        self.ny = ny
        
        # step size
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        # relaxation parameter
        self.omega = omega
        
        # create history vectors
        self.Xh = self.da4.createGlobalVec()
        self.Xp = self.da4.createGlobalVec()
        self.Ph = self.da1.createGlobalVec()
        self.Pp = self.da1.createGlobalVec()
        
        # create local vectors
        self.localB  = da4.createLocalVec()
        self.localX  = da4.createLocalVec()
        self.localXh = da4.createLocalVec()
        self.localXp = da4.createLocalVec()
        self.localPh = da1.createLocalVec()
        self.localPp = da1.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETSc_MHD_Derivatives(nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X, Vec P):
        x  = self.da4.getVecArray(X)
        p  = self.da1.getVecArray(P)
        xh = self.da4.getVecArray(self.Xh)
        ph = self.da1.getVecArray(self.Ph)
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        xh[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        ph[xs:xe, ys:ye]    = p[xs:xe, ys:ye]
        
    
    def update_previous(self, Vec X, Vec P):
        x  = self.da4.getVecArray(X)
        p  = self.da1.getVecArray(P)
        xp = self.da4.getVecArray(self.Xp)
        pp = self.da1.getVecArray(self.Pp)
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        xp[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        pp[xs:xe, ys:ye]    = p[xs:xe, ys:ye]
        
    
#    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(self.Xh, self.localXh)
        self.da4.globalToLocal(self.Xp, self.localXp)
        
        xh = self.da4.getVecArray(self.localXh)
        xp = self.da4.getVecArray(self.localXp)
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxp = xp[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byp = xp[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxp = xp[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyp = xp[...][:,:,3]
        
        
        cdef np.float64_t fac_fdudx = 0.25 / 32. / self.hx
        cdef np.float64_t fac_fdudy = 0.25 / 32. / self.hy
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                
                # B_x
                row.index = (i,j)
                row.field = 0
                
                # fdudx(Vxp, Bx)
                # fdudx(Vxh, Bx)
                # fdudy(Vyp, Bx)
                # fdudy(Vyh, Bx)
                col.field = 0

                for index, value in [
                        ((i-1, j-1), - 1. * ( Vxp[i-1, j  ] + Vxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxp[i,   j  ] + Vxp[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Vxp[i-1, j-1] + Vxp[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Vxp[i-1, j  ] + Vxp[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i-1, j+1] + Vxp[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Vxp[i-1, j  ] + Vxp[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Vxp[i,   j  ] + Vxp[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Vxp[i+1, j-1] - Vxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Vxp[i+1, j-1] - Vxp[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j+1] - Vxp[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j+1] - Vxp[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Vxp[i,   j-1] + Vxp[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Vxp[i+1, j-1] + Vxp[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Vxp[i,   j-1] + Vxp[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Vxp[i,   j  ] + Vxp[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Vxp[i,   j+1] + Vxp[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Vxp[i,   j  ] + Vxp[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Vxp[i+1, j  ] + Vxp[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Vxh[i-1, j  ] + Vxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxh[i,   j  ] + Vxh[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Vxh[i-1, j-1] + Vxh[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Vxh[i-1, j  ] + Vxh[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i-1, j+1] + Vxh[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Vxh[i-1, j  ] + Vxh[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Vxh[i,   j  ] + Vxh[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Vxh[i+1, j-1] - Vxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Vxh[i+1, j-1] - Vxh[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j+1] - Vxh[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j+1] - Vxh[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Vxh[i,   j-1] + Vxh[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Vxh[i+1, j-1] + Vxh[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Vxh[i,   j-1] + Vxh[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Vxh[i,   j  ] + Vxh[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Vxh[i,   j+1] + Vxh[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Vxh[i,   j  ] + Vxh[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Vxh[i+1, j  ] + Vxh[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Vyp[i-1, j-1] + Vyp[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Vyp[i-1, j+1] - Vyp[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Vyp[i-1, j  ] + Vyp[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Vyp[i-1, j-1] + Vyp[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j-1] + Vyp[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Vyp[i-1, j+1] - Vyp[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j+1] - Vyp[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Vyp[i-1, j  ] + Vyp[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i+1, j  ] + Vyp[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j-1] + Vyp[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j+1] - Vyp[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i+1, j  ] + Vyp[i+1, j+1]) * fac_fdudy),
                        ((i-1, j-1), - 1. * ( Vyh[i-1, j-1] + Vyh[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Vyh[i-1, j+1] - Vyh[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Vyh[i-1, j  ] + Vyh[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Vyh[i-1, j-1] + Vyh[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j-1] + Vyh[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Vyh[i-1, j+1] - Vyh[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j+1] - Vyh[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Vyh[i-1, j  ] + Vyh[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i+1, j  ] + Vyh[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j-1] + Vyh[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j+1] - Vyh[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i+1, j  ] + Vyh[i+1, j+1]) * fac_fdudy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
                
                # fdudx(Bxp, Vx)
                # fdudx(Bxh, Vx)
                # fdudy(Byp, Vx)
                # fdudy(Byh, Vx)
                col.field = 2
                
                for index, value in [
                        ((i-1, j-1), - 1. * ( Bxp[i-1, j  ] + Bxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxp[i,   j  ] + Bxp[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Bxp[i-1, j-1] + Bxp[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Bxp[i-1, j  ] + Bxp[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i-1, j+1] + Bxp[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Bxp[i-1, j  ] + Bxp[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Bxp[i,   j  ] + Bxp[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Bxp[i+1, j-1] - Bxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Bxp[i+1, j-1] - Bxp[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j+1] - Bxp[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j+1] - Bxp[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Bxp[i,   j-1] + Bxp[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Bxp[i+1, j-1] + Bxp[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Bxp[i,   j-1] + Bxp[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Bxp[i,   j  ] + Bxp[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Bxp[i,   j+1] + Bxp[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Bxp[i,   j  ] + Bxp[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Bxp[i+1, j  ] + Bxp[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Bxh[i-1, j  ] + Bxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxh[i,   j  ] + Bxh[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Bxh[i-1, j-1] + Bxh[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Bxh[i-1, j  ] + Bxh[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i-1, j+1] + Bxh[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Bxh[i-1, j  ] + Bxh[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Bxh[i,   j  ] + Bxh[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Bxh[i+1, j-1] - Bxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Bxh[i+1, j-1] - Bxh[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j+1] - Bxh[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j+1] - Bxh[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Bxh[i,   j-1] + Bxh[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Bxh[i+1, j-1] + Bxh[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Bxh[i,   j-1] + Bxh[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Bxh[i,   j  ] + Bxh[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Bxh[i,   j+1] + Bxh[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Bxh[i,   j  ] + Bxh[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Bxh[i+1, j  ] + Bxh[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Byp[i-1, j-1] + Byp[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Byp[i-1, j+1] - Byp[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Byp[i-1, j  ] + Byp[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Byp[i-1, j-1] + Byp[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j-1] + Byp[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Byp[i-1, j+1] - Byp[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j+1] - Byp[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Byp[i-1, j  ] + Byp[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i+1, j  ] + Byp[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j-1] + Byp[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j+1] - Byp[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i+1, j  ] + Byp[i+1, j+1]) * fac_fdudy),
                        ((i-1, j-1), - 1. * ( Byh[i-1, j-1] + Byh[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Byh[i-1, j+1] - Byh[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Byh[i-1, j  ] + Byh[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Byh[i-1, j-1] + Byh[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j-1] + Byh[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Byh[i-1, j+1] - Byh[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j+1] - Byh[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Byh[i-1, j  ] + Byh[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i+1, j  ] + Byh[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j-1] + Byh[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j+1] - Byh[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i+1, j  ] + Byh[i+1, j+1]) * fac_fdudy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
                
                # B_y
                row.index = (i,j)
                row.field = 1
                
                # fdudx(Vxp, By)
                # fdudx(Vxh, By)
                # fdudy(Vyp, By)
                # fdudy(Vyh, By)
                col.field = 1
                
                for index, value in [
                        ((i-1, j-1), - 1. * ( Vxp[i-1, j  ] + Vxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxp[i,   j  ] + Vxp[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Vxp[i-1, j-1] + Vxp[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Vxp[i-1, j  ] + Vxp[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i-1, j+1] + Vxp[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Vxp[i-1, j  ] + Vxp[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Vxp[i,   j  ] + Vxp[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Vxp[i+1, j-1] - Vxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Vxp[i+1, j-1] - Vxp[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j+1] - Vxp[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j+1] - Vxp[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Vxp[i,   j-1] + Vxp[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Vxp[i+1, j-1] + Vxp[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Vxp[i,   j-1] + Vxp[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Vxp[i,   j  ] + Vxp[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Vxp[i,   j+1] + Vxp[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Vxp[i,   j  ] + Vxp[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Vxp[i+1, j  ] + Vxp[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Vxh[i-1, j  ] + Vxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxh[i,   j  ] + Vxh[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Vxh[i-1, j-1] + Vxh[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Vxh[i-1, j  ] + Vxh[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i-1, j+1] + Vxh[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Vxh[i-1, j  ] + Vxh[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Vxh[i,   j  ] + Vxh[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Vxh[i+1, j-1] - Vxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Vxh[i+1, j-1] - Vxh[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j+1] - Vxh[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j+1] - Vxh[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Vxh[i,   j-1] + Vxh[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Vxh[i+1, j-1] + Vxh[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Vxh[i,   j-1] + Vxh[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Vxh[i,   j  ] + Vxh[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Vxh[i,   j+1] + Vxh[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Vxh[i,   j  ] + Vxh[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Vxh[i+1, j  ] + Vxh[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Vyp[i-1, j-1] + Vyp[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Vyp[i-1, j+1] - Vyp[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Vyp[i-1, j  ] + Vyp[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Vyp[i-1, j-1] + Vyp[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j-1] + Vyp[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Vyp[i-1, j+1] - Vyp[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j+1] - Vyp[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Vyp[i-1, j  ] + Vyp[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i+1, j  ] + Vyp[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j-1] + Vyp[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j+1] - Vyp[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i+1, j  ] + Vyp[i+1, j+1]) * fac_fdudy),
                        ((i-1, j-1), - 1. * ( Vyh[i-1, j-1] + Vyh[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Vyh[i-1, j+1] - Vyh[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Vyh[i-1, j  ] + Vyh[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Vyh[i-1, j-1] + Vyh[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j-1] + Vyh[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Vyh[i-1, j+1] - Vyh[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j+1] - Vyh[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Vyh[i-1, j  ] + Vyh[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i+1, j  ] + Vyh[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j-1] + Vyh[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j+1] - Vyh[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i+1, j  ] + Vyh[i+1, j+1]) * fac_fdudy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
                
                # fdudx(Bxp, Vy)
                # fdudx(Bxh, Vy)
                # fdudy(Byp, Vy)
                # fdudy(Byh, Vy)
                col.field = 3
                
                for index, value in [
                        ((i-1, j-1), - 1. * ( Bxp[i-1, j  ] + Bxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxp[i,   j  ] + Bxp[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Bxp[i-1, j-1] + Bxp[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Bxp[i-1, j  ] + Bxp[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i-1, j+1] + Bxp[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Bxp[i-1, j  ] + Bxp[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Bxp[i,   j  ] + Bxp[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Bxp[i+1, j-1] - Bxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Bxp[i+1, j-1] - Bxp[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j+1] - Bxp[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j+1] - Bxp[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Bxp[i,   j-1] + Bxp[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Bxp[i+1, j-1] + Bxp[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Bxp[i,   j-1] + Bxp[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Bxp[i,   j  ] + Bxp[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Bxp[i,   j+1] + Bxp[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Bxp[i,   j  ] + Bxp[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Bxp[i+1, j  ] + Bxp[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Bxh[i-1, j  ] + Bxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxh[i,   j  ] + Bxh[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Bxh[i-1, j-1] + Bxh[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Bxh[i-1, j  ] + Bxh[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i-1, j+1] + Bxh[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Bxh[i-1, j  ] + Bxh[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Bxh[i,   j  ] + Bxh[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Bxh[i+1, j-1] - Bxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Bxh[i+1, j-1] - Bxh[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j+1] - Bxh[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j+1] - Bxh[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Bxh[i,   j-1] + Bxh[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Bxh[i+1, j-1] + Bxh[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Bxh[i,   j-1] + Bxh[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Bxh[i,   j  ] + Bxh[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Bxh[i,   j+1] + Bxh[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Bxh[i,   j  ] + Bxh[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Bxh[i+1, j  ] + Bxh[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Byp[i-1, j-1] + Byp[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Byp[i-1, j+1] - Byp[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Byp[i-1, j  ] + Byp[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Byp[i-1, j-1] + Byp[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j-1] + Byp[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Byp[i-1, j+1] - Byp[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j+1] - Byp[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Byp[i-1, j  ] + Byp[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i+1, j  ] + Byp[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j-1] + Byp[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j+1] - Byp[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i+1, j  ] + Byp[i+1, j+1]) * fac_fdudy),
                        ((i-1, j-1), - 1. * ( Byh[i-1, j-1] + Byh[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Byh[i-1, j+1] - Byh[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Byh[i-1, j  ] + Byh[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Byh[i-1, j-1] + Byh[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j-1] + Byh[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Byh[i-1, j+1] - Byh[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j+1] - Byh[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Byh[i-1, j  ] + Byh[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i+1, j  ] + Byh[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j-1] + Byh[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j+1] - Byh[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i+1, j  ] + Byh[i+1, j+1]) * fac_fdudy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
                
                # V_x
                row.index = (i,j)
                row.field = 2
                
                # fdudx(Vxp, Vx)
                # fdudx(Vxh, Vx)
                # fdudy(Vyp, Vx)
                # fdudy(Vyh, Vx)
                col.field = 2
                
                for index, value in [
                        ((i-1, j-1), - 1. * ( Vxp[i-1, j  ] + Vxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxp[i,   j  ] + Vxp[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Vxp[i-1, j-1] + Vxp[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Vxp[i-1, j  ] + Vxp[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i-1, j+1] + Vxp[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Vxp[i-1, j  ] + Vxp[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Vxp[i,   j  ] + Vxp[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Vxp[i+1, j-1] - Vxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Vxp[i+1, j-1] - Vxp[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j+1] - Vxp[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j+1] - Vxp[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Vxp[i,   j-1] + Vxp[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Vxp[i+1, j-1] + Vxp[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Vxp[i,   j-1] + Vxp[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Vxp[i,   j  ] + Vxp[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Vxp[i,   j+1] + Vxp[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Vxp[i,   j  ] + Vxp[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Vxp[i+1, j  ] + Vxp[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Vxh[i-1, j  ] + Vxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxh[i,   j  ] + Vxh[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Vxh[i-1, j-1] + Vxh[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Vxh[i-1, j  ] + Vxh[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i-1, j+1] + Vxh[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Vxh[i-1, j  ] + Vxh[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Vxh[i,   j  ] + Vxh[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Vxh[i+1, j-1] - Vxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Vxh[i+1, j-1] - Vxh[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j+1] - Vxh[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j+1] - Vxh[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Vxh[i,   j-1] + Vxh[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Vxh[i+1, j-1] + Vxh[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Vxh[i,   j-1] + Vxh[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Vxh[i,   j  ] + Vxh[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Vxh[i,   j+1] + Vxh[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Vxh[i,   j  ] + Vxh[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Vxh[i+1, j  ] + Vxh[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Vyp[i-1, j-1] + Vyp[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Vyp[i-1, j+1] - Vyp[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Vyp[i-1, j  ] + Vyp[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Vyp[i-1, j-1] + Vyp[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j-1] + Vyp[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Vyp[i-1, j+1] - Vyp[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j+1] - Vyp[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Vyp[i-1, j  ] + Vyp[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i+1, j  ] + Vyp[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j-1] + Vyp[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j+1] - Vyp[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i+1, j  ] + Vyp[i+1, j+1]) * fac_fdudy),
                        ((i-1, j-1), - 1. * ( Vyh[i-1, j-1] + Vyh[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Vyh[i-1, j+1] - Vyh[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Vyh[i-1, j  ] + Vyh[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Vyh[i-1, j-1] + Vyh[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j-1] + Vyh[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Vyh[i-1, j+1] - Vyh[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j+1] - Vyh[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Vyh[i-1, j  ] + Vyh[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i+1, j  ] + Vyh[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j-1] + Vyh[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j+1] - Vyh[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i+1, j  ] + Vyh[i+1, j+1]) * fac_fdudy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
                
                # fdudx(Bxp, Bx)
                # fdudx(Bxh, Bx)
                # fdudy(Byp, Bx)
                # fdudy(Byh, Bx)
                col.field = 0
                
                for index, value in [
                        ((i-1, j-1), - 1. * ( Bxp[i-1, j  ] + Bxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxp[i,   j  ] + Bxp[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Bxp[i-1, j-1] + Bxp[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Bxp[i-1, j  ] + Bxp[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i-1, j+1] + Bxp[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Bxp[i-1, j  ] + Bxp[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Bxp[i,   j  ] + Bxp[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Bxp[i+1, j-1] - Bxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Bxp[i+1, j-1] - Bxp[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j+1] - Bxp[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j+1] - Bxp[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Bxp[i,   j-1] + Bxp[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Bxp[i+1, j-1] + Bxp[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Bxp[i,   j-1] + Bxp[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Bxp[i,   j  ] + Bxp[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Bxp[i,   j+1] + Bxp[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Bxp[i,   j  ] + Bxp[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Bxp[i+1, j  ] + Bxp[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Bxh[i-1, j  ] + Bxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxh[i,   j  ] + Bxh[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Bxh[i-1, j-1] + Bxh[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Bxh[i-1, j  ] + Bxh[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i-1, j+1] + Bxh[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Bxh[i-1, j  ] + Bxh[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Bxh[i,   j  ] + Bxh[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Bxh[i+1, j-1] - Bxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Bxh[i+1, j-1] - Bxh[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j+1] - Bxh[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j+1] - Bxh[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Bxh[i,   j-1] + Bxh[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Bxh[i+1, j-1] + Bxh[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Bxh[i,   j-1] + Bxh[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Bxh[i,   j  ] + Bxh[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Bxh[i,   j+1] + Bxh[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Bxh[i,   j  ] + Bxh[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Bxh[i+1, j  ] + Bxh[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Byp[i-1, j-1] + Byp[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Byp[i-1, j+1] - Byp[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Byp[i-1, j  ] + Byp[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Byp[i-1, j-1] + Byp[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j-1] + Byp[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Byp[i-1, j+1] - Byp[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j+1] - Byp[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Byp[i-1, j  ] + Byp[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i+1, j  ] + Byp[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j-1] + Byp[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j+1] - Byp[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i+1, j  ] + Byp[i+1, j+1]) * fac_fdudy),
                        ((i-1, j-1), - 1. * ( Byh[i-1, j-1] + Byh[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Byh[i-1, j+1] - Byh[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Byh[i-1, j  ] + Byh[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Byh[i-1, j-1] + Byh[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j-1] + Byh[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Byh[i-1, j+1] - Byh[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j+1] - Byh[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Byh[i-1, j  ] + Byh[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i+1, j  ] + Byh[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j-1] + Byh[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j+1] - Byh[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i+1, j  ] + Byh[i+1, j+1]) * fac_fdudy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
                
                # V_y
                row.index = (i,j)
                row.field = 3
                
                # fdudx(Vxp, Vy)
                # fdudx(Vxh, Vy)
                # fdudy(Vyp, Vy)
                # fdudy(Vyh, Vy)
                col.field = 3
                
                for index, value in [
                        ((i-1, j-1), - 1. * ( Vxp[i-1, j  ] + Vxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxp[i,   j  ] + Vxp[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Vxp[i-1, j-1] + Vxp[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Vxp[i-1, j  ] + Vxp[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i-1, j+1] + Vxp[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Vxp[i-1, j  ] + Vxp[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Vxp[i,   j  ] + Vxp[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Vxp[i+1, j-1] - Vxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Vxp[i+1, j-1] - Vxp[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j+1] - Vxp[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Vxp[i+1, j  ] - Vxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxp[i+1, j+1] - Vxp[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Vxp[i,   j-1] + Vxp[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Vxp[i+1, j-1] + Vxp[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Vxp[i,   j-1] + Vxp[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Vxp[i,   j  ] + Vxp[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Vxp[i,   j+1] + Vxp[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Vxp[i,   j  ] + Vxp[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Vxp[i+1, j  ] + Vxp[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Vxh[i-1, j  ] + Vxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxh[i,   j  ] + Vxh[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Vxh[i-1, j-1] + Vxh[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Vxh[i-1, j  ] + Vxh[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i-1, j+1] + Vxh[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Vxh[i-1, j  ] + Vxh[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Vxh[i,   j  ] + Vxh[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Vxh[i+1, j-1] - Vxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Vxh[i+1, j-1] - Vxh[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j+1] - Vxh[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Vxh[i+1, j  ] - Vxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Vxh[i+1, j+1] - Vxh[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Vxh[i,   j-1] + Vxh[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Vxh[i+1, j-1] + Vxh[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Vxh[i,   j-1] + Vxh[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Vxh[i,   j  ] + Vxh[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Vxh[i,   j+1] + Vxh[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Vxh[i,   j  ] + Vxh[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Vxh[i+1, j  ] + Vxh[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Vyp[i-1, j-1] + Vyp[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Vyp[i-1, j+1] - Vyp[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Vyp[i-1, j  ] + Vyp[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Vyp[i-1, j-1] + Vyp[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j-1] + Vyp[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Vyp[i-1, j+1] - Vyp[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j+1] - Vyp[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Vyp[i-1, j  ] + Vyp[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i+1, j  ] + Vyp[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Vyp[i,   j-1] + Vyp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j-1] + Vyp[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Vyp[i,   j+1] - Vyp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyp[i+1, j+1] - Vyp[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Vyp[i,   j  ] + Vyp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyp[i+1, j  ] + Vyp[i+1, j+1]) * fac_fdudy),
                        ((i-1, j-1), - 1. * ( Vyh[i-1, j-1] + Vyh[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Vyh[i-1, j+1] - Vyh[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Vyh[i-1, j  ] + Vyh[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Vyh[i-1, j-1] + Vyh[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j-1] + Vyh[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Vyh[i-1, j+1] - Vyh[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j+1] - Vyh[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Vyh[i-1, j  ] + Vyh[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i+1, j  ] + Vyh[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Vyh[i,   j-1] + Vyh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j-1] + Vyh[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Vyh[i,   j+1] - Vyh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Vyh[i+1, j+1] - Vyh[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Vyh[i,   j  ] + Vyh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Vyh[i+1, j  ] + Vyh[i+1, j+1]) * fac_fdudy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
                
                # fdudx(Bxp, By)
                # fdudx(Bxh, By)
                # fdudy(Byp, By)
                # fdudy(Byh, By)
                col.field = 1
                
                for index, value in [
                        ((i-1, j-1), - 1. * ( Bxp[i-1, j  ] + Bxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxp[i,   j  ] + Bxp[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Bxp[i-1, j-1] + Bxp[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Bxp[i-1, j  ] + Bxp[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i-1, j+1] + Bxp[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Bxp[i-1, j  ] + Bxp[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Bxp[i,   j  ] + Bxp[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Bxp[i+1, j-1] - Bxp[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Bxp[i+1, j-1] - Bxp[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j+1] - Bxp[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Bxp[i+1, j  ] - Bxp[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxp[i+1, j+1] - Bxp[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Bxp[i,   j-1] + Bxp[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Bxp[i+1, j-1] + Bxp[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Bxp[i,   j-1] + Bxp[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Bxp[i,   j  ] + Bxp[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Bxp[i,   j+1] + Bxp[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Bxp[i,   j  ] + Bxp[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Bxp[i+1, j  ] + Bxp[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Bxh[i-1, j  ] + Bxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxh[i,   j  ] + Bxh[i,   j-1]) * fac_fdudx),
                        ((i-1, j  ), - 1. * ( Bxh[i-1, j-1] + Bxh[i,   j-1]) * fac_fdudx \
                                     - 2. * ( Bxh[i-1, j  ] + Bxh[i,   j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i-1, j+1] + Bxh[i,   j+1]) * fac_fdudx),
                        ((i-1, j+1), - 1. * ( Bxh[i-1, j  ] + Bxh[i-1, j+1]) * fac_fdudx \
                                     - 1. * ( Bxh[i,   j  ] + Bxh[i,   j+1]) * fac_fdudx),
                        ((i,   j-1), - 1. * ( Bxh[i+1, j-1] - Bxh[i-1, j-1]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx),
                        ((i,   j  ), - 1. * ( Bxh[i+1, j-1] - Bxh[i-1, j-1]) * fac_fdudx \
                                     - 2. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j+1] - Bxh[i-1, j+1]) * fac_fdudx),
                        ((i,   j+1), - 1. * ( Bxh[i+1, j  ] - Bxh[i-1, j  ]) * fac_fdudx \
                                     - 1. * ( Bxh[i+1, j+1] - Bxh[i-1, j+1]) * fac_fdudx),
                        ((i+1, j-1), + 1. * ( Bxh[i,   j-1] + Bxh[i,   j  ]) * fac_fdudx \
                                     + 1. * ( Bxh[i+1, j-1] + Bxh[i+1, j  ]) * fac_fdudx),
                        ((i+1, j  ), + 1. * ( Bxh[i,   j-1] + Bxh[i+1, j-1]) * fac_fdudx \
                                     + 2. * ( Bxh[i,   j  ] + Bxh[i+1, j  ]) * fac_fdudx \
                                     + 1. * ( Bxh[i,   j+1] + Bxh[i+1, j+1]) * fac_fdudx),
                        ((i+1, j+1), + 1. * ( Bxh[i,   j  ] + Bxh[i,   j+1]) * fac_fdudx \
                                     + 1. * ( Bxh[i+1, j  ] + Bxh[i+1, j+1]) * fac_fdudx),
                        ((i-1, j-1), - 1. * ( Byp[i-1, j-1] + Byp[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Byp[i-1, j+1] - Byp[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Byp[i-1, j  ] + Byp[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Byp[i-1, j-1] + Byp[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j-1] + Byp[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Byp[i-1, j+1] - Byp[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j+1] - Byp[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Byp[i-1, j  ] + Byp[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i+1, j  ] + Byp[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Byp[i,   j-1] + Byp[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j-1] + Byp[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Byp[i,   j+1] - Byp[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byp[i+1, j+1] - Byp[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Byp[i,   j  ] + Byp[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byp[i+1, j  ] + Byp[i+1, j+1]) * fac_fdudy),
                        ((i-1, j-1), - 1. * ( Byh[i-1, j-1] + Byh[i-1, j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy),
                        ((i-1, j  ), - 1. * ( Byh[i-1, j+1] - Byh[i-1, j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy),
                        ((i-1, j+1), + 1. * ( Byh[i-1, j  ] + Byh[i-1, j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy),
                        ((i,   j-1), - 1. * ( Byh[i-1, j-1] + Byh[i-1, j  ]) * fac_fdudy \
                                     - 2. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j-1] + Byh[i+1, j  ]) * fac_fdudy),
                        ((i,   j  ), - 1. * ( Byh[i-1, j+1] - Byh[i-1, j-1]) * fac_fdudy \
                                     - 2. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j+1] - Byh[i+1, j-1]) * fac_fdudy),
                        ((i,   j+1), + 1. * ( Byh[i-1, j  ] + Byh[i-1, j+1]) * fac_fdudy \
                                     + 2. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i+1, j  ] + Byh[i+1, j+1]) * fac_fdudy),
                        ((i+1, j-1), - 1. * ( Byh[i,   j-1] + Byh[i,   j  ]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j-1] + Byh[i+1, j  ]) * fac_fdudy),
                        ((i+1, j  ), - 1. * ( Byh[i,   j+1] - Byh[i,   j-1]) * fac_fdudy \
                                     - 1. * ( Byh[i+1, j+1] - Byh[i+1, j-1]) * fac_fdudy),
                        ((i+1, j+1), + 1. * ( Byh[i,   j  ] + Byh[i,   j+1]) * fac_fdudy \
                                     + 1. * ( Byh[i+1, j  ] + Byh[i+1, j+1]) * fac_fdudy),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
                
        A.assemble()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Matrix")
        
                
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(self.Xh, self.localXh)
        self.da4.globalToLocal(self.Xp, self.localXp)
        self.da1.globalToLocal(self.Ph, self.localPh)
        self.da1.globalToLocal(self.Pp, self.localPp)
        
        xh = self.da4.getVecArray(self.localXh)
        xp = self.da4.getVecArray(self.localXp)
        ph = self.da1.getVecArray(self.localPh)
        pp = self.da1.getVecArray(self.localPp)
        
        cdef np.ndarray[np.float64_t, ndim=3] b = self.da4.getVecArray(B)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxh = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxh = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyh = xh[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bxp = xp[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Byp = xp[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vxp = xp[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vyp = xp[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Ph  = ph[...]
        cdef np.ndarray[np.float64_t, ndim=2] Pp  = pp[...]
        
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                # B_x
                b[iy, jy, 0] = self.derivatives.dt(Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxp, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxh, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyp, Bxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyh, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxp, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxh, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byp, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byh, Vxh, ix, jx)
                
                # B_y
                b[iy, jy, 1] = self.derivatives.dt(Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxp, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxh, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyp, Byh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyh, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxp, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byp, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byh, Vyh, ix, jx)
                
                # V_x
                b[iy, jy, 2] = self.derivatives.dt(Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxp, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxh, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyp, Vxh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyh, Vxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxp, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxh, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byp, Bxh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byh, Bxh, ix, jx) \
                             - 0.5 * self.derivatives.gradx(Pp, ix, jx) \
                             - 0.5 * self.derivatives.gradx(Ph, ix, jx)

                # V_y
                b[iy, jy, 3] = self.derivatives.dt(Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxp, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudx(Vxh, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyp, Vyh, ix, jx) \
                             - 0.25 * self.derivatives.fdudy(Vyh, Vyh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxp, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudx(Bxh, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byp, Byh, ix, jx) \
                             + 0.25 * self.derivatives.fdudy(Byh, Byh, ix, jx) \
                             - 0.5 * self.derivatives.grady(Pp, ix, jx) \
                             - 0.5 * self.derivatives.grady(Ph, ix, jx)
                
    
    
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     RHS")
    
    
#    @cython.boundscheck(False)
    def pressure(self, Vec X, Vec Y):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da4.globalToLocal(X, self.localX)
        
        x = self.da4.getVecArray(self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=2] P  = self.da1.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Pp = self.da1.getVecArray(self.Pp)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By = x[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[...][:,:,3]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                P[iy, jy] = Pp[iy, jy] \
                          - self.omega * ( \
                             + self.derivatives.gradx(Vx, ix, jx) \
                             + self.derivatives.grady(Vy, ix, jx) \
                          )

    
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Pressure")



#    @cython.boundscheck(False)
    def timestep(self, np.ndarray[np.float64_t, ndim=3] x,
                       np.ndarray[np.float64_t, ndim=2] p,
                       np.ndarray[np.float64_t, ndim=3] y):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] Bx = x[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] By = x[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Vx = x[:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Vy = x[:,:,3]
        cdef np.ndarray[np.float64_t, ndim=2] P  = p[:,:]
        
        
        for j in np.arange(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
                # B_x
                y[iy, jy, 0] = \
                             - self.derivatives.fdudx(Vx, Bx, ix, jx) \
                             - self.derivatives.fdudy(Vy, Bx, ix, jx) \
                             + self.derivatives.fdudx(Bx, Vx, ix, jx) \
                             + self.derivatives.fdudy(By, Vx, ix, jx)
                    
                # B_y
                y[iy, jy, 1] = \
                             - self.derivatives.fdudx(Vx, By, ix, jx) \
                             - self.derivatives.fdudy(Vy, By, ix, jx) \
                             + self.derivatives.fdudx(Bx, Vy, ix, jx) \
                             + self.derivatives.fdudy(By, Vy, ix, jx)
                                
                # V_x
                y[iy, jy, 2] = \
                             - self.derivatives.fdudx(Vx, Vx, ix, jx) \
                             - self.derivatives.fdudy(Vy, Vx, ix, jx) \
                             + self.derivatives.fdudx(Bx, Bx, ix, jx) \
                             + self.derivatives.fdudy(By, Bx, ix, jx) \
                             - self.derivatives.gradx(P, ix, jx)
                              
                # V_y
                y[iy, jy, 3] = \
                             - self.derivatives.fdudx(Vx, Vy, ix, jx) \
                             - self.derivatives.fdudy(Vy, Vy, ix, jx) \
                             + self.derivatives.fdudx(Bx, By, ix, jx) \
                             + self.derivatives.fdudy(By, By, ix, jx) \
                             - self.derivatives.grady(P, ix, jx)
          
