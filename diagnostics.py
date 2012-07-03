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
        
        self.ht = self.tGrid[2] - self.tGrid[1]
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
        
        self.B  = np.zeros((self.nx, self.ny))
        self.V  = np.zeros((self.nx, self.ny))
        
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
        
        
        self.read_from_hdf5(0)
        self.update_invariants(0)
        
        
        
    def read_from_hdf5(self, iTime):
        self.Bx = self.hdf5['Bx'][iTime,:,:]
        self.By = self.hdf5['By'][iTime,:,:]
        self.Vx = self.hdf5['Vx'][iTime,:,:]
        self.Vy = self.hdf5['Vy'][iTime,:,:]
        
        self.B = np.sqrt( self.Bx**2 + self.By**2 )
        self.V = np.sqrt( self.Vx**2 + self.Vy**2 )
        
    
    def update_invariants(self, iTime):
        
#        self.E_magnetic = 0.5 * (self.Bx**2 + self.By**2).sum() * self.hx * self.hy
#        self.E_velocity = 0.5 * (self.Vx**2 + self.Vy**2).sum() * self.hx * self.hy
        
        self.E_magnetic = 0.0
        self.E_velocity = 0.0
        
        for ix in range(0, self.nx):
            ixp = (ix+1) % self.nx
            
            for iy in range(0, self.ny):
                iyp = (iy+1) % self.ny
                
                self.E_magnetic = (self.Bx[ix,iy] + self.Bx[ixp,iy] + self.Bx[ixp,iyp] + self.Bx[ix,iyp])**2 \
                                + (self.By[ix,iy] + self.By[ixp,iy] + self.By[ixp,iyp] + self.By[ix,iyp])**2
                                
                self.E_velocity = (self.Vx[ix,iy] + self.Vx[ixp,iy] + self.Vx[ixp,iyp] + self.Vx[ix,iyp])**2 \
                                + (self.Vy[ix,iy] + self.Vy[ixp,iy] + self.Vy[ixp,iyp] + self.Vy[ix,iyp])**2

        self.E_magnetic *= 0.5 * self.hx * self.hy / 16.
        self.E_velocity *= 0.5 * self.hx * self.hy / 16.
        
        
        self.L1_magnetic = self.B.sum() * self.hx * self.hy
        self.L1_velocity = self.V.sum() * self.hx * self.hy
        
        self.L2_magnetic = 0.5 * (self.B**2).sum() * self.hx * self.hy
        self.L2_velocity = 0.5 * (self.V**2).sum() * self.hx * self.hy
        
        self.energy   = self.E_magnetic + self.E_velocity 
        self.helicity = (self.Bx * self.Vx + self.By * self.Vy).sum() * self.hx * self.hy
    
        
        if iTime == 0:
            self.E0 = self.energy
            self.H0 = self.helicity
            
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
            self.E_error = (self.energy   - self.E0) / self.E0
            self.H_error = (self.helicity - self.H0)
            
            self.L1_magnetic_error = (self.L1_magnetic - self.L1_magnetic_0) / self.L1_magnetic_0
            self.L1_velocity_error = (self.L1_velocity - self.L1_velocity_0) / self.L1_velocity_0
            self.L2_magnetic_error = (self.L2_magnetic - self.L2_magnetic_0) / self.L2_magnetic_0
            self.L2_velocity_error = (self.L2_velocity - self.L2_velocity_0) / self.L2_velocity_0
        
    
