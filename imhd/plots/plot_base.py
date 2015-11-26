'''
Created on Nov 26, 2015

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter


class PlotMHD2Dbase(object):
    '''
    Basic plotting class creating data structures and implementing
    data reading functions. 
    '''

    def __init__(self, diagnostics, nTime=0, nPlot=1, ntMax=0, write=False, prefix='viIMHD2D_'):
        '''
        Set some control variables and create all data arrays.
        '''
        
        self.write  = write
        self.prefix = prefix
        
        if ntMax == 0:
            ntMax = diagnostics.nt
        
        if nTime > 0 and nTime < ntMax:
            self.nTime = nTime
        else:
            self.nTime = ntMax
        
        self.nPlot = nPlot
        
        self.diagnostics = diagnostics
        
        self.update_boundaries()
        
        
        self.E_velocity  = np.zeros_like(diagnostics.tGrid)
        self.E_magnetic  = np.zeros_like(diagnostics.tGrid)
        
        self.energy      = np.zeros_like(diagnostics.tGrid)
        self.helicity    = np.zeros_like(diagnostics.tGrid)
        
        
        self.x = np.zeros(diagnostics.nx+1)
        self.y = np.zeros(diagnostics.ny+1)
        
        self.x[0:-1] = self.diagnostics.xGrid
        self.x[  -1] = self.x[-2] + self.diagnostics.hx
        
        self.y[0:-1] = self.diagnostics.yGrid
        self.y[  -1] = self.y[-2] + self.diagnostics.hy
        
        self.Bx      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.By      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Bix     = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Biy     = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Vx      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Vy      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.P       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.PB      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        self.A       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Ai      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.B       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Bi      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.J       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.V       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        self.divB    = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.divV    = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        
    def load_data(self):
        '''
        Copy data from diagnostics accounting for periodic boundaries.
        '''
    
        self.A[0:-1, 0:-1] = self.diagnostics.A[:,:]
        self.A[  -1, 0:-1] = self.diagnostics.A[0,:]
        self.A[   :,   -1] = self.A[:,0]
        
        self.J[0:-1, 0:-1] = self.diagnostics.J[:,:]
        self.J[  -1, 0:-1] = self.diagnostics.J[0,:]
        self.J[   :,   -1] = self.J[:,0]
        
        self.B [0:-1, 0:-1] = self.diagnostics.B [:,:]
        self.B [  -1, 0:-1] = self.diagnostics.B [0,:]
        self.B [   :,   -1] = self.B[:,0]
        
        self.Bx[0:-1, 0:-1] = self.diagnostics.Bx[:,:]
        self.Bx[  -1, 0:-1] = self.diagnostics.Bx[0,:]
        self.Bx[   :,   -1] = self.Bx[:,0]
        
        self.By[0:-1, 0:-1] = self.diagnostics.By[:,:]
        self.By[  -1, 0:-1] = self.diagnostics.By[0,:]
        self.By[   :,   -1] = self.By[:,0]
        
        self.divB[0:-1, 0:-1] = self.diagnostics.divB[:,:]
        self.divB[  -1, 0:-1] = self.diagnostics.divB[0,:]
        self.divB[   :,   -1] = self.divB[:,0]
        
        self.V [0:-1, 0:-1] = self.diagnostics.V [:,:]
        self.V [  -1, 0:-1] = self.diagnostics.V [0,:]
        self.V [   :,   -1] = self.V[:,0]
        
        self.Vx[0:-1, 0:-1] = self.diagnostics.Vx[:,:]
        self.Vx[  -1, 0:-1] = self.diagnostics.Vx[0,:]
        self.Vx[   :,   -1] = self.Vx[:,0]
        
        self.Vy[0:-1, 0:-1] = self.diagnostics.Vy[:,:]
        self.Vy[  -1, 0:-1] = self.diagnostics.Vy[0,:]
        self.Vy[   :,   -1] = self.Vy[:,0]
        
        self.divV[0:-1, 0:-1] = self.diagnostics.divV[:,:]
        self.divV[  -1, 0:-1] = self.diagnostics.divV[0,:]
        self.divV[   :,   -1] = self.divV[:,0]
        
        self.P[0:-1, 0:-1] = self.diagnostics.P[:,:]
        self.P[  -1, 0:-1] = self.diagnostics.P[0,:]
        self.P[   :,   -1] = self.P[:,0]
        
        self.PB[0:-1, 0:-1] = self.diagnostics.e_magnetic[:,:]
        self.PB[  -1, 0:-1] = self.diagnostics.e_magnetic[0,:]
        self.PB[   :,   -1] = self.PB[:,0]
        
        
        if self.diagnostics.inertial_mhd:
            self.Ai[0:-1, 0:-1] = self.diagnostics.Ai[:,:]
            self.Ai[  -1, 0:-1] = self.diagnostics.Ai[0,:]
            self.Ai[   :,   -1] = self.Ai[:,0]
            
            self.Bi[0:-1, 0:-1] = self.diagnostics.Bi[:,:]
            self.Bi[  -1, 0:-1] = self.diagnostics.Bi[0,:]
            self.Bi[   :,   -1] = self.Bi[:,0]
            
            self.Bix[0:-1, 0:-1] = self.diagnostics.Bix[:,:]
            self.Bix[  -1, 0:-1] = self.diagnostics.Bix[0,:]
            self.Bix[   :,   -1] = self.Bix[:,0]
            
            self.Biy[0:-1, 0:-1] = self.diagnostics.Biy[:,:]
            self.Biy[  -1, 0:-1] = self.diagnostics.Biy[0,:]
            self.Biy[   :,   -1] = self.Biy[:,0]
        

    def update_boundaries(self):
        self.update_boundaries_magnetic()
        self.update_boundaries_velocity()
        self.update_boundaries_pressure()
        self.update_boundaries_potential()
        
        
    def update_boundaries_magnetic(self):
        Bmin = min(self.diagnostics.Bx.min(), self.diagnostics.By.min(), -self.diagnostics.Bx.max(), -self.diagnostics.By.max())
        Bmax = max(self.diagnostics.Bx.max(), self.diagnostics.By.max(), -self.diagnostics.Bx.min(), -self.diagnostics.By.min())
        
        self.BxTicks = np.linspace(Bmin, Bmax, 11, endpoint=True)
        self.ByTicks = np.linspace(Bmin, Bmax, 11, endpoint=True)
        
        # TODO add Bix and Biy

        
    def update_boundaries_velocity(self):
        Vmin = 2. * min(self.diagnostics.Vx.min(), self.diagnostics.Vy.min(), -self.diagnostics.Vx.max(), -self.diagnostics.Vy.max())
        Vmax = 2. * max(self.diagnostics.Vx.max(), self.diagnostics.Vy.max(), -self.diagnostics.Vx.min(), -self.diagnostics.Vy.min())
        
        divVmin = self.diagnostics.divV.min()
        divVmax = self.diagnostics.divV.max()
        
        if Vmin == Vmax:
            Vmin -= 1.
            Vmax += 1.
        
        if divVmin == divVmax:
            divVmin -= 0.1
            divVmax += 0.1
        
        self.VxTicks = np.linspace(Vmin, Vmax, 11, endpoint=True)
        self.VyTicks = np.linspace(Vmin, Vmax, 11, endpoint=True)

        self.divVTicks = np.linspace(divVmin, divVmax, 11, endpoint=True)
    
    
    def update_boundaries_pressure(self):
        Pmin = min(self.diagnostics.P.min(), -self.diagnostics.P.max())
        Pmax = max(self.diagnostics.P.max(), -self.diagnostics.P.min())
        
        if Pmin == Pmax:
            Pmin -= 1.
            Pmax += 1.
        
        self.PTicks = np.linspace(Pmin, Pmax, 11, endpoint=True)

        
    def update_boundaries_potential(self):
        Amin = min(self.diagnostics.A.min(), -self.diagnostics.A.max())
        Amax = max(self.diagnostics.A.max(), -self.diagnostics.A.min())
        Adif = Amax - Amin
        
#        self.ATicks = np.linspace(Amin + 0.3 * Adif, Amax + 0.2 * Adif, 21, endpoint=True)
        self.ATicks = np.linspace(Amin + 0.01 * Adif, Amax - 0.01 * Adif, 31)
        
        # TODO add Ai
        
    
    def get_timerange(self, iTime):
        '''
        Obtain timetrace plotting range corresponding to current time index.
        '''
        
        tStart = iTime - (self.nTime+1)
        tEnd   = iTime
        
        if tStart < 0:
            tStart = 0
        
        xStart = self.diagnostics.tGrid[tStart]
        xEnd   = self.diagnostics.tGrid[tStart+self.nTime]
        
        return tStart, tEnd, xStart, xEnd
    
    
    def update(self, iTime, final=False, draw=True):
        pass


    def update_sub(self, iTime):
        pass
