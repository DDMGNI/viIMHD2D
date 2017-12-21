'''
Created on Jul 2, 2012

@author: mkraus
'''

from cpython cimport bool

import h5py
import numpy as np
from math import sqrt


cdef class Diagnostics(object):
    '''
    classdocs
    '''

    cdef object hdf5

    cdef public bool inertial_mhd
    
    cdef public double[:] tGrid
    cdef public double[:] xGrid
    cdef public double[:] yGrid
    
    cdef public double ht, hx, hy
    cdef public double Lx, Ly
    
    cdef public int nt, nx, ny, n
    
    cdef public double tMin, tMax
    cdef public double xMin, xMax
    cdef public double yMin, yMax

    cdef public double[:,:] Bx
    cdef public double[:,:] By
    cdef public double[:,:] Vx
    cdef public double[:,:] Vy
    cdef public double[:,:] Bix
    cdef public double[:,:] Biy
    cdef public double[:,:] P
    cdef public double[:,:] A
    cdef public double[:,:] J
    cdef public double[:,:] B
    cdef public double[:,:] V
    cdef public double[:,:] Ai
    cdef public double[:,:] Bi
    cdef public double[:,:] divB
    cdef public double[:,:] divV
    
    cdef public double[:,:] e_magnetic
    cdef public double[:,:] e_velocity

    cdef public double E_magnetic
    cdef public double E_velocity
    
    cdef public double energy
    cdef public double helicity
    cdef public double magnetic
    
    cdef public double L1_magnetic
    cdef public double L1_velocity
    cdef public double L2_magnetic
    cdef public double L2_velocity
    cdef public double L2_A
    cdef public double L2_X
    
    cdef public double E0
    cdef public double H0
    cdef public double M0
    
    cdef public double L1_magnetic_0
    cdef public double L1_velocity_0
    cdef public double L2_magnetic_0
    cdef public double L2_velocity_0
    cdef public double L2_A_0
    cdef public double L2_X_0
    
    cdef public double E_error
    cdef public double H_error
    cdef public double M_error
    
    cdef public double L1_magnetic_error
    cdef public double L1_velocity_error
    cdef public double L2_magnetic_error
    cdef public double L2_velocity_error
    cdef public double L2_A_error
    cdef public double L2_X_error
    
    cdef public bool plot_energy
    cdef public bool plot_helicity
    cdef public bool plot_magnetic
    cdef public bool plot_L2_A
    cdef public bool plot_L2_X
    
    
    
    def __init__(self):
        print("Hello 0")
        pass
    
    
    def __init__(self, hdf5_file):
        '''
        Constructor
        '''

        print("Hello 1")
        self.hdf5 = h5py.File(hdf5_file, 'r')
        
        print("Hello 2")
        assert self.hdf5 != None
        
        print("Hello 3")
        if 'Bix' in self.hdf5 and 'Biy' in self.hdf5: 
            self.inertial_mhd = True
        else:
            self.inertial_mhd = False
        
        
        print("Hello 4")
        self.tGrid = self.hdf5['t'][:].flatten()
        
        self.xGrid = self.hdf5['x'][:]
        self.yGrid = self.hdf5['y'][:]
        
        if len(self.tGrid) > 1:
            self.ht = self.tGrid[1] - self.tGrid[0]
        else:
            self.ht = 0
        
        self.nt = len(self.tGrid)-1
        
        self.Lx = (self.xGrid[-1] - self.xGrid[0]) + (self.xGrid[1] - self.xGrid[0])
        self.Ly = (self.yGrid[-1] - self.yGrid[0]) + (self.yGrid[1] - self.yGrid[0])
        
        self.nx = len(self.xGrid)
        self.ny = len(self.yGrid)
        self.n  = self.nx * self.ny
        
        self.hx = self.xGrid[1] - self.xGrid[0]
        self.hy = self.yGrid[1] - self.yGrid[0]
        
        self.tMin = self.tGrid[ 0]
        self.tMax = self.tGrid[-1]
        self.xMin = self.xGrid[ 0]
        self.xMax = self.xGrid[-1]
        self.yMin = self.yGrid[ 0]
        self.yMax = self.yGrid[-1]
        
        
        print("nt = %i (%i)" % (self.nt, len(self.tGrid)) )
        print("nx = %i" % (self.nx))
        print("ny = %i" % (self.ny))
        print
        print("ht = %f" % (self.ht))
        print("hx = %f" % (self.hx))
        print("hy = %f" % (self.hy))
        print
        print("tGrid:")
        print(np.array(self.tGrid))
        print
        print("xGrid:")
        print(np.array(self.xGrid))
        print
        print("yGrid:")
        print(np.array(self.yGrid))
        print
        
        
        self.Bx = np.zeros((self.nx, self.ny))
        self.By = np.zeros((self.nx, self.ny))
        self.Vx = np.zeros((self.nx, self.ny))
        self.Vy = np.zeros((self.nx, self.ny))
#         self.Bix = None
#         self.Biy = None
        
        self.P  = np.zeros((self.nx, self.ny))
        
        self.e_magnetic = np.zeros((self.nx, self.ny))
        self.e_velocity = np.zeros((self.nx, self.ny))
        
        self.A  = np.zeros((self.nx, self.ny))
        self.J  = np.zeros((self.nx, self.ny))
        self.B  = np.zeros((self.nx, self.ny))
        self.V  = np.zeros((self.nx, self.ny))
        self.Ai = np.zeros((self.nx, self.ny))
        self.Bi = np.zeros((self.nx, self.ny))
        
        self.divB = np.zeros((self.nx, self.ny))
        self.divV = np.zeros((self.nx, self.ny))
        
        self.E_magnetic  = 0.0
        self.E_velocity  = 0.0
        
        self.energy   = 0.0
        self.helicity = 0.0
        self.magnetic = 0.0
        
        self.L1_magnetic = 0.0
        self.L1_velocity = 0.0
        self.L2_magnetic = 0.0
        self.L2_velocity = 0.0
        self.L2_A        = 0.0
        self.L2_X        = 0.0
        
        self.E0       = 0.0
        self.H0       = 0.0
        self.M0       = 0.0
        
        self.L1_magnetic_0 = 0.0
        self.L1_velocity_0 = 0.0
        self.L2_magnetic_0 = 0.0
        self.L2_velocity_0 = 0.0
        self.L2_A_0        = 0.0
        self.L2_X_0        = 0.0
        
        self.E_error  = 0.0
        self.H_error  = 0.0
        self.M_error  = 0.0
        
        self.L1_magnetic_error = 0.0
        self.L1_velocity_error = 0.0
        self.L2_magnetic_error = 0.0
        self.L2_velocity_error = 0.0
        self.L2_A_error        = 0.0
        self.L2_X_error        = 0.0
        
        self.plot_energy   = False
        self.plot_helicity = False
        self.plot_magnetic = False
        self.plot_L2_A = False
        self.plot_L2_X = False
        
        self.read_from_hdf5(0)
        self.update_invariants(0)
        
        
		
    cdef kahan_sum1(self, double[:,:] X):
        cdef int i, j
        cdef double r = 0 # result
        cdef double t = 0 # temporary
        cdef double e = 0 # error
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                e = e + X[i,j]
                t = r
                r = t + e
                e = e + (t - r)
            
        return r
    
		
    cdef kahan_sum2(self, double[:,:] X, double[:,:] Y):
        cdef int i, j
        cdef double r = 0 # result
        cdef double t = 0 # temporary
        cdef double e = 0 # error
    
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                e = e + X[i,j] * Y[i,j]
                t = r
                r = t + e
                e = e + (t - r)
        
        return r
    
		
    cdef kahan_sum_square(self, double[:,:] X):
        cdef int i, j
        cdef double r = 0 # result
        cdef double t = 0 # temporary
        cdef double e = 0 # error
    
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                e = e + X[i,j]**2
                t = r
                r = t + e
                e = e + (t - r)
        
        return r
    
    
    cdef my_abs(self, double[:,:] X, double[:,:] Y, double[:,:] A):
        cdef int i, j
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i,j] = sqrt(X[i,j]**2 + Y[i,j]**2)
    
    
    cdef remove_average(self, double[:,:] X):
        cdef int i, j
        cdef double x = 0
        cdef double a
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x += X[i,j]
                
        a = x / (X.shape[0] * X.shape[1])
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i,j] -= a
		
        
    def read_from_hdf5(self, iTime):
        self.Bx = self.hdf5['Bx'][iTime,:,:].T
        self.By = self.hdf5['By'][iTime,:,:].T
        self.Vx = self.hdf5['Vx'][iTime,:,:].T
        self.Vy = self.hdf5['Vy'][iTime,:,:].T

        if self.inertial_mhd:
            self.Bix = self.hdf5['Bix'][iTime,:,:].T
            self.Biy = self.hdf5['Biy'][iTime,:,:].T
        
#        self.P  = self.hdf5['P'][iTime,:,:].T
        
        self.my_abs(self.Bx, self.By, self.B)
        self.my_abs(self.Vx, self.Vy, self.V)
        
        if self.inertial_mhd:
            self.my_abs(self.Bix, self.Biy, self.Bi)
        
        self.remove_average(self.P)
        
    
    def update_invariants(self, iTime):
        cdef int i, j, ix, ixm, ixp, iy, iym, iyp
        
        # reconstruction of vector potential
        self.A[-1,-1] = 0.0
        
        for i in range(0, self.nx):
            ix = self.nx-i-1
            ixm = (ix-1+self.nx) % self.nx
            ixp = (ix+1+self.nx) % self.nx
            
            if  i < self.nx-1:
                self.A[ix,0] = self.A[ixp,0] + self.hx * self.By[ix,0]
#                self.A[ix,0] = self.A[ixp,0] + self.hx * (self.By[ixm,0] + 4*self.By[ix,0] + self.By[ixp,0])/6.
        
            for j in range(0, self.ny):
                iy = self.ny-j-1
                iym = (iy-1+self.ny) % self.ny
                iyp = (iy+1+self.ny) % self.ny
                
                self.A[ix,iy] = self.A[ix,iyp] - self.hy * self.Bx[ix,iy]  
#                self.A[ix,iy] = self.A[ix,iyp] - self.hy * (self.Bx[ix,iym] + 4.*self.Bx[ix,iy] + self.Bx[ix,iyp])/6.
            
#         self.A -= self.A.mean()
#         self.A -= self.A.min()
        
        
        # reconstruction of generalised vector potential
        if self.inertial_mhd:
            self.Ai[-1,-1] = 0.0
            
            for i in range(self.nx):
                ix = self.nx-i-1
                ixm = (ix-1+self.nx) % self.nx
                ixp = (ix+1+self.nx) % self.nx
                
                if  i < self.nx-1:
                    self.Ai[ix,0] = self.Ai[ixp,0] + self.hx * self.Biy[ix,0]
            
                for j in range(self.ny):
                    iy = self.ny-j-1
                    iym = (iy-1+self.ny) % self.ny
                    iyp = (iy+1+self.ny) % self.ny
                    
                    self.Ai[ix,iy] = self.Ai[ix,iyp] - self.hy * self.Bix[ix,iy]  
                
#             self.Ai -= self.Ai.mean()
        
        
        # reconstruction of current density
        for ix in range(0, self.nx):
            ixm = (ix-1 + self.nx) % self.nx
            
            for iy in range(0, self.ny):
                iym = (iy-1 + self.ny) % self.ny
                
                self.J[ix,iy] = (self.By[ixm,iy] - self.By[ix,iy]) / self.hx \
                              - (self.Bx[ix,iym] - self.Bx[ix,iy]) / self.hy
        
        
        if self.inertial_mhd:
            self.E_magnetic = 0.5 * (self.kahan_sum2(self.Bx, self.Bix) + self.kahan_sum2(self.By, self.Biy)) * self.hx * self.hy
            self.E_velocity = 0.5 * (self.kahan_sum_square(self.Vx) + self.kahan_sum_square(self.Vy)) * self.hx * self.hy
            self.helicity   = (self.kahan_sum2(self.Bix, self.Vx) + self.kahan_sum2(self.Biy, self.Vy)) * self.hx * self.hy
            self.L2_X       = 0.5 * self.kahan_sum_square(self.Ai) * self.hx * self.hy
            self.magnetic   = self.kahan_sum1(self.Ai) * self.hx * self.hy
        else:
            self.E_magnetic = 0.5 * (self.kahan_sum_square(self.Bx) + self.kahan_sum_square(self.By)) * self.hx * self.hy
            self.E_velocity = 0.5 * (self.kahan_sum_square(self.Vx) + self.kahan_sum_square(self.Vy)) * self.hx * self.hy
            self.helicity   = (self.kahan_sum2(self.Bx, self.Vx) + self.kahan_sum2(self.By, self.Vy)) * self.hx * self.hy
            self.magnetic   = self.kahan_sum1(self.A) * self.hx * self.hy
        
        
        self.L1_magnetic = self.kahan_sum1(self.B) * self.hx * self.hy
        self.L1_velocity = self.kahan_sum1(self.V) * self.hx * self.hy
        
        self.L2_magnetic = 0.5 * self.kahan_sum_square(self.B) * self.hx * self.hy
        self.L2_velocity = 0.5 * self.kahan_sum_square(self.V) * self.hx * self.hy
        self.L2_A        = 0.5 * self.kahan_sum_square(self.A) * self.hx * self.hy
        
        self.energy   = self.E_magnetic + self.E_velocity 

        
        if iTime == 0:
            self.E0 = self.energy
            self.H0 = self.helicity
            self.M0 = self.magnetic
            
            self.L1_magnetic_0 = self.L1_magnetic
            self.L1_velocity_0 = self.L1_velocity 
            self.L2_magnetic_0 = self.L2_magnetic
            self.L2_velocity_0 = self.L2_velocity
            self.L2_A_0        = self.L2_A
            self.L2_X_0        = self.L2_X
            
            if np.abs(self.E0) < 1E-15:
                self.plot_energy = True
            
            if np.abs(self.H0) < 1E-15:
                self.plot_helicity = True
            
            if np.abs(self.M0) < 1E-15:
                self.plot_magnetic = True
            
            if np.abs(self.L2_A_0) < 1E-15:
                self.plot_L2_A = True
            
            if np.abs(self.L2_X_0) < 1E-15:
                self.plot_L2_X = True
        
            self.E_error  = 0.0
            self.H_error  = 0.0
            self.M_error  = 0.0
            
            self.L1_magnetic_error = 0.0
            self.L1_velocity_error = 0.0
            self.L2_magnetic_error = 0.0
            self.L2_velocity_error = 0.0
        
        else:
            if self.plot_energy:
                self.E_error = (self.energy)
            else:
                self.E_error = (self.energy   - self.E0) / self.E0
                
            if self.plot_helicity:
                self.H_error = (self.helicity)
            else:
                self.H_error = (self.helicity - self.H0) / self.H0
            
            if self.plot_magnetic:
                self.M_error = (self.magnetic)
            else:
                self.M_error = (self.magnetic - self.M0) / self.M0
            
            if self.plot_L2_A:
                self.L2_A_error = (self.L2_A)
            else:
                self.L2_A_error = (self.L2_A - self.L2_A_0) / self.L2_A_0
            
            if self.plot_L2_X:
                self.L2_X_error = (self.L2_X)
            else:
                self.L2_X_error = (self.L2_X - self.L2_X_0) / self.L2_X_0
            
            self.L1_magnetic_error = (self.L1_magnetic - self.L1_magnetic_0) #/ self.L1_magnetic_0
            self.L1_velocity_error = (self.L1_velocity - self.L1_velocity_0) #/ self.L1_velocity_0
            self.L2_magnetic_error = (self.L2_magnetic - self.L2_magnetic_0) #/ self.L2_magnetic_0
            self.L2_velocity_error = (self.L2_velocity - self.L2_velocity_0) #/ self.L2_velocity_0
        
        
    
    def calculate_divergence(self):
        cdef int i, j, ix, ixm, ixp, iy, iym, iyp
        
        for ix in range(0, self.nx):
            ixp = (ix+1+self.nx) % self.nx
            
            for iy in range(0, self.ny):
                iyp = (iy+1+self.nx) % self.ny
                
                self.divB[ix,iy] = (self.Bx[ixp,iy] - self.Bx[ix,iy]) / self.hx \
                                 + (self.By[ix,iyp] - self.By[ix,iy]) / self.hy
                
                self.divV[ix,iy] = (self.Vx[ixp,iy] - self.Vx[ix,iy]) / self.hx \
                                 + (self.Vy[ix,iyp] - self.Vy[ix,iy]) / self.hy
                
