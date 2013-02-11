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
        
        
        self.tGrid = self.hdf5['t'][:,0,0]
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
        
        self.tMin = self.tGrid[ 1]
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
        
        self.P  = np.zeros((self.nx, self.ny))
        self.J  = np.zeros((self.nx, self.ny))
        
        self.e_magnetic = np.zeros((self.nx, self.ny))
        self.e_velocity = np.zeros((self.nx, self.ny))
        
        self.A  = np.zeros((self.nx, self.ny))
        self.B  = np.zeros((self.nx, self.ny))
        self.V  = np.zeros((self.nx, self.ny))
        
        self.divB = np.zeros((self.nx, self.ny))
        self.divV = np.zeros((self.nx, self.ny))
        
        self.E_magnetic  = 0.0
        self.E_velocity  = 0.0
        
        self.energy   = 0.0
        self.helicity = 0.0
        
        self.L1_magnetic = 0.0
        self.L1_velocity = 0.0
        self.L2_magnetic = 0.0
        self.L2_velocity = 0.0
        
        self.E0       = 0.0
        self.H0       = 0.0
        
        self.L1_magnetic_0 = 0.0
        self.L1_velocity_0 = 0.0
        self.L2_magnetic_0 = 0.0
        self.L2_velocity_0 = 0.0
        
        self.E_error  = 0.0
        self.H_error  = 0.0
        
        self.L1_magnetic_error = 0.0
        self.L1_velocity_error = 0.0
        self.L2_magnetic_error = 0.0
        self.L2_velocity_error = 0.0
        
        self.plot_energy   = False
        self.plot_helicity = False
        
        self.read_from_hdf5(0)
        self.update_invariants(0)
        
        
        
    def read_from_hdf5(self, iTime):
        self.Bx = self.hdf5['Bx'][iTime,:,:].T
        self.By = self.hdf5['By'][iTime,:,:].T
        self.Vx = self.hdf5['Vx'][iTime,:,:].T
        self.Vy = self.hdf5['Vy'][iTime,:,:].T
        
        self.P  = self.hdf5['P'][iTime,:,:].T
        
        self.B = np.sqrt( self.Bx**2 + self.By**2 )
        self.V = np.sqrt( self.Vx**2 + self.Vy**2 )
        
        
        Pave = self.P.sum() / (self.nx * self.ny)
        self.P[:,:] -= Pave 
        
    
    def update_invariants(self, iTime):
        
        self.A[-1,-1] = 0.0
        
        for i in range(0, self.nx):
            ix = self.nx-i-1
            ixm = (ix-1+self.nx) % self.nx
            
            for j in range(0, self.ny):
                iy = self.ny-j-1
                iyp = (iy+1+self.ny) % self.ny
                
                self.A[ix,iy] = self.A[ix,iyp] - self.hy * self.Bx[ix,iy]  
            
            if ixm > 0:
                self.A[ixm,iy] = self.A[ix,iy] + self.hx * self.By[ix,iy]
        
#        self.A -= self.A.mean()
        self.A -= self.A.min()
        
        
        self.E_magnetic = 0.5 * (self.Bx**2 + self.By**2).sum() * self.hx * self.hy
        self.E_velocity = 0.5 * (self.Vx**2 + self.Vy**2).sum() * self.hx * self.hy
        
        
        for ix in range(0, self.nx):
            ixm = (ix-1 + self.nx) % self.nx
            
            for iy in range(0, self.ny):
                iym = (iy-1 + self.ny) % self.ny
                
                self.e_magnetic[ix,iy] = (self.Bx[ix,iy] + self.Bx[ix,iym])**2 \
                                       + (self.By[ix,iy] + self.By[ixm,iy])**2
                                
                self.e_velocity[ix,iy] = (self.Vx[ix,iy] + self.Vx[ix,iym])**2 \
                                       + (self.Vy[ix,iy] + self.Vy[ixm,iy])**2
        
        self.e_magnetic *= 0.5 * 0.25
        self.e_velocity *= 0.5 * 0.25
        
        
#        self.E_magnetic = 0.0
#        self.E_velocity = 0.0
#        
#        for ix in range(0, self.nx):
#            ixp = (ix+1) % self.nx
#            
#            for iy in range(0, self.ny):
#                iyp = (iy+1) % self.ny
#                
#                self.E_magnetic += (self.Bx[ix,iy]**2 + self.Bx[ixp,iy]**2 + self.Bx[ixp,iyp]**2 + self.Bx[ix,iyp]**2) \
#                                +  (self.By[ix,iy]**2 + self.By[ixp,iy]**2 + self.By[ixp,iyp]**2 + self.By[ix,iyp]**2)
#                                
#                self.E_velocity += (self.Vx[ix,iy]**2 + self.Vx[ixp,iy]**2 + self.Vx[ixp,iyp]**2 + self.Vx[ix,iyp]**2) \
#                                +  (self.Vy[ix,iy]**2 + self.Vy[ixp,iy]**2 + self.Vy[ixp,iyp]**2 + self.Vy[ix,iyp]**2)
#
#        self.E_magnetic *= 0.5 * self.hx * self.hy / 4.
#        self.E_velocity *= 0.5 * self.hx * self.hy / 4.


#        self.E_magnetic = 0.0
#        self.E_velocity = 0.0
#        
#        for ix in range(0, self.nx):
#            ixm = (ix-1 + self.nx) % self.nx
#            
#            for iy in range(0, self.ny):
#                iym = (iy-1 + self.ny) % self.ny
#                
#                self.E_magnetic += (self.Bx[ix,iy] + self.Bx[ix,iym])**2 \
#                                +  (self.By[ix,iy] + self.By[ixm,iy])**2
#                                
#                self.E_velocity += (self.Vx[ix,iy] + self.Vx[ix,iym])**2 \
#                                +  (self.Vy[ix,iy] + self.Vy[ixm,iy])**2
#
#        self.E_magnetic *= 0.25 * 0.5 * self.hx * self.hy
#        self.E_velocity *= 0.25 * 0.5 * self.hx * self.hy
        
        
        self.L1_magnetic = self.B.sum() * self.hx * self.hy
        self.L1_velocity = self.V.sum() * self.hx * self.hy
        
        self.L2_magnetic = 0.5 * (self.B**2).sum() * self.hx * self.hy
        self.L2_velocity = 0.5 * (self.V**2).sum() * self.hx * self.hy
        
        self.energy   = self.E_magnetic + self.E_velocity 
        self.helicity = (self.Bx * self.Vx + self.By * self.Vy).sum() * self.hx * self.hy
    
        
        if iTime == 0:
            self.E0 = self.energy
            self.H0 = self.helicity
            
#            if self.E0 == 0:
            if np.abs(self.E0) < 1E-15:
                self.plot_energy = True
            
#            if self.H0 == 0:
            if np.abs(self.H0) < 1E-15:
                self.plot_helicity = True
            
            
            self.L1_magnetic_0 = self.L1_magnetic
            self.L1_velocity_0 = self.L1_velocity 
            self.L2_magnetic_0 = self.L2_magnetic
            self.L2_velocity_0 = self.L2_velocity
        
            self.E_error  = 0.0
            self.H_error  = 0.0
            
            self.L1_magnetic_error = 0.0
            self.L1_velocity_error = 0.0
            self.L2_magnetic_error = 0.0
            self.L2_velocity_error = 0.0
        
        else:
            if self.E0 == 0:
                self.E_error = (self.energy   - self.E0)
            else:
                self.E_error = (self.energy   - self.E0) / self.E0
                
            if self.H0 == 0:
                self.H_error = (self.helicity - self.H0)
            else:
                self.H_error = (self.helicity - self.H0) / self.H0
            
            
            self.L1_magnetic_error = (self.L1_magnetic - self.L1_magnetic_0) #/ self.L1_magnetic_0
            self.L1_velocity_error = (self.L1_velocity - self.L1_velocity_0) #/ self.L1_velocity_0
            self.L2_magnetic_error = (self.L2_magnetic - self.L2_magnetic_0) #/ self.L2_magnetic_0
            self.L2_velocity_error = (self.L2_velocity - self.L2_velocity_0) #/ self.L2_velocity_0
        
        
        # current
        for ix in range(0, self.nx):
            ixm = (ix-1 + self.nx) % self.nx
            
            for iy in range(0, self.ny):
                iym = (iy-1 + self.ny) % self.ny
                
                self.J[ix,iy] = (self.By[ixm,iy] - self.By[ix,iy]) / self.hx \
                              - (self.Bx[ix,iym] - self.Bx[ix,iy]) / self.hy
                                
        
        
        
        
    
    def calculate_divergence(self):
        for ix in range(0, self.nx):
            ixp = (ix+1+self.nx) % self.nx
            
            for iy in range(0, self.ny):
                iyp = (iy+1+self.nx) % self.ny
                
                self.divB[ix,iy] = (self.Bx[ixp,iy] - self.Bx[ix,iy]) / self.hx \
                                 + (self.By[ix,iyp] - self.By[ix,iy]) / self.hy
                
                self.divV[ix,iy] = (self.Vx[ixp,iy] - self.Vx[ix,iy]) / self.hx \
                                 + (self.Vy[ix,iyp] - self.Vy[ix,iy]) / self.hy
                
