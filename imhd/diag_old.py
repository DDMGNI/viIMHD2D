'''
Created on Jul 2, 2012

@author: mkraus
'''

import h5py
import numpy as np


class Diagnostics(object):
    '''
    classdocs
    '''


    def __init__(self, hdf5_file):
        '''
        Constructor
        '''

        self.hdf5 = h5py.File(hdf5_file, 'r')
        
        assert self.hdf5 != None
        
        if 'Bix' in self.hdf5 and 'Biy' in self.hdf5: 
            self.inertial_mhd = True
        else:
            self.inertial_mhd = False
        
        
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
        print(self.tGrid)
        print
        print("xGrid:")
        print(self.xGrid)
        print
        print("yGrid:")
        print(self.yGrid)
        print
        
        
        self.Bx = np.zeros((self.nx, self.ny))
        self.By = np.zeros((self.nx, self.ny))
        self.Vx = np.zeros((self.nx, self.ny))
        self.Vy = np.zeros((self.nx, self.ny))
        self.Bix = None
        self.Biy = None
        
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
        
        
        
    def read_from_hdf5(self, iTime):
        self.Bx = self.hdf5['Bx'][iTime,:,:].T
        self.By = self.hdf5['By'][iTime,:,:].T
        self.Vx = self.hdf5['Vx'][iTime,:,:].T
        self.Vy = self.hdf5['Vy'][iTime,:,:].T

        if self.inertial_mhd:
            self.Bix = self.hdf5['Bix'][iTime,:,:].T
            self.Biy = self.hdf5['Biy'][iTime,:,:].T
        
#        self.P  = self.hdf5['P'][iTime,:,:].T
        
        self.B = np.sqrt( self.Bx**2 + self.By**2 )
        self.V = np.sqrt( self.Vx**2 + self.Vy**2 )
        
        if self.inertial_mhd:
            self.Bi = np.sqrt( self.Bix**2 + self.Biy**2 )
        
        Pave = self.P.sum() / (self.nx * self.ny)
        self.P[:,:] -= Pave 
        
    
    def update_invariants(self, iTime):
        
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
            self.E_magnetic = 0.5 * (self.Bx*self.Bix + self.By*self.Biy).sum() * self.hx * self.hy
            self.E_velocity = 0.5 * (self.Vx**2 + self.Vy**2).sum() * self.hx * self.hy
            self.helicity   = (self.Bix * self.Vx + self.Biy * self.Vy).sum() * self.hx * self.hy
            self.L2_X       = 0.5 * (self.Ai**2).sum() * self.hx * self.hy
            self.magnetic   = self.Ai.sum() * self.hx * self.hy
        else:
            self.E_magnetic = 0.5 * (self.Bx**2 + self.By**2).sum() * self.hx * self.hy
            self.E_velocity = 0.5 * (self.Vx**2 + self.Vy**2).sum() * self.hx * self.hy
            self.helicity   = (self.Bx * self.Vx + self.By * self.Vy).sum() * self.hx * self.hy
            self.magnetic   = self.A.sum() * self.hx * self.hy
        
        
        self.L1_magnetic = self.B.sum() * self.hx * self.hy
        self.L1_velocity = self.V.sum() * self.hx * self.hy
        
        self.L2_magnetic = 0.5 * (self.B**2).sum() * self.hx * self.hy
        self.L2_velocity = 0.5 * (self.V**2).sum() * self.hx * self.hy

        self.L2_A        = 0.5 * (self.A**2).sum() * self.hx * self.hy
        
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
        for ix in range(0, self.nx):
            ixp = (ix+1+self.nx) % self.nx
            
            for iy in range(0, self.ny):
                iyp = (iy+1+self.nx) % self.ny
                
                self.divB[ix,iy] = (self.Bx[ixp,iy] - self.Bx[ix,iy]) / self.hx \
                                 + (self.By[ix,iyp] - self.By[ix,iy]) / self.hy
                
                self.divV[ix,iy] = (self.Vx[ixp,iy] - self.Vx[ix,iy]) / self.hx \
                                 + (self.Vy[ix,iyp] - self.Vy[ix,iy]) / self.hy
                
