'''
Created on Jul 02, 2012

@author: mkraus
'''

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm, colors, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter


class PlotMHD2D(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, nTime=0, nPlot=1):
        '''
        Constructor
        '''
        
        self.nrows = 2
        self.ncols = 4
        
        if nTime > 0 and nTime < diagnostics.nt:
            self.nTime = nTime
        else:
            self.nTime = diagnostics.nt
        
        self.iTime = 0
        self.nPlot = nPlot
        
        self.diagnostics = diagnostics
        
        
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
        self.Vx      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Vy      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.P       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        self.PB      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.J       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
                
        self.B       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.V       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
#        self.read_data()
        self.update_boundaries()
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=2.5, wspace=0.25)
#        plt.subplots_adjust(hspace=0.2, wspace=0.25)
        plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = 0.0' % (diagnostics.tGrid[self.iTime]), horizontalalignment='center') 
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        ## -> limit to 1.1f precision
        majorFormatter.set_powerlimits((-1,+1))
        majorFormatter.set_scientific(True)

        # add data for zero timepoint
        self.add_timepoint()
        
        # set up plots
        self.axes  = {}
        self.conts = {}
        self.cbars = {}
        self.lines = {}
        self.vecs  = {}
        
        
        # create subplots
        gs = gridspec.GridSpec(20, 5)
        self.gs = gs
        
        self.axes["Bx"]    = plt.subplot(gs[ 0: 4,  0])
        self.axes["By"]    = plt.subplot(gs[ 4: 8,  0])
        self.axes["Vx"]    = plt.subplot(gs[ 8:12,  0])
        self.axes["Vy"]    = plt.subplot(gs[12:16,  0])
        self.axes["P"]     = plt.subplot(gs[16:20,  0])
        self.axes["PB"]    = plt.subplot(gs[ 0:10,1:3])
        self.axes["J"]     = plt.subplot(gs[10:20,1:3])
        self.axes["Emag"]  = plt.subplot(gs[ 0: 5,3:5])
        self.axes["Evel"]  = plt.subplot(gs[ 5:10,3:5])
        self.axes["E"]     = plt.subplot(gs[10:15,3:5])
        self.axes["H"]     = plt.subplot(gs[15:20,3:5])
        
        
        self.axes["Bx"].set_title('$B_{x} (x,y)$')
        self.axes["By"].set_title('$B_{y} (x,y)$')
        self.axes["Vx"].set_title('$V_{x} (x,y)$')
        self.axes["Vy"].set_title('$V_{y} (x,y)$')
        self.axes["P" ].set_title('$P (x,y)$')
        self.axes["PB"].set_title('$B^{2} (x,y)$')
        self.axes["J" ].set_title('$J (x,y)$')
        
        
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, self.BxTicks, cmap=cm.jet)
        self.cbars["Bx"] = self.figure.colorbar(self.conts["Bx"], ax=self.axes["Bx"], orientation='vertical', ticks=self.BxTicks)#, format='%0.2E'
       
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, self.ByTicks, cmap=cm.jet)
        self.cbars["By"] = self.figure.colorbar(self.conts["By"], ax=self.axes["By"], orientation='vertical', ticks=self.ByTicks)
        
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, self.VxTicks, cmap=cm.jet)
        self.cbars["Vx"] = self.figure.colorbar(self.conts["Vx"], ax=self.axes["Vx"], orientation='vertical', ticks=self.VxTicks)
        
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, self.VyTicks, cmap=cm.jet)
        self.cbars["Vy"] = self.figure.colorbar(self.conts["Vy"], ax=self.axes["Vy"], orientation='vertical', ticks=self.VyTicks)
        
        self.conts["P" ] = self.axes["P"].contourf(self.x, self.y, self.P.T, self.PTicks, cmap=cm.jet)
        self.cbars["P" ] = self.figure.colorbar(self.conts["P"], ax=self.axes["P"], orientation='vertical', ticks=self.PTicks)
        
#        self.conts["PB"] = self.axes["PB"].contourf(self.x, self.y, self.PB.T, ticks=self.PBTicks)
#        self.cbars["PB"] = self.figure.colorbar(self.conts["PB"], ax=self.axes["PB"], orientation='vertical', ticks=self.PBTicks)
        self.conts["PB"] = self.axes["PB"].contourf(self.x, self.y, self.PB.T, 51, norm=self.PBnorm)
#        self.cbars["PB"] = self.figure.colorbar(self.conts["PB"], ax=self.axes["PB"], orientation='vertical')

#        self.conts["J" ] = self.axes["J"].contourf(self.x, self.y, self.J.T, ticks=self.JTicks)
#        self.cbars["J" ] = self.figure.colorbar(self.conts["J"], ax=self.axes["J"], orientation='vertical', ticks=self.JTicks)
        self.conts["J" ] = self.axes["J"].contourf(self.x, self.y, self.J.T, 51, norm=self.Jnorm)
#        self.cbars["J" ] = self.figure.colorbar(self.conts["J"], ax=self.axes["J"], orientation='vertical')

        
        tStart, tEnd, xStart, xEnd = self.get_timerange()

        self.lines["Emag" ], = self.axes["Emag" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.E_magnetic [tStart:tEnd])
        self.lines["Evel" ], = self.axes["Evel" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.E_velocity [tStart:tEnd])
        self.lines["E"    ], = self.axes["E"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.energy     [tStart:tEnd])
        self.lines["H"    ], = self.axes["H"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.helicity   [tStart:tEnd])
        
        self.axes["Emag"].set_title('$E_{B} (t)$')
        self.axes["Evel"].set_title('$E_{V} (t)$')
        
        if self.diagnostics.plot_energy:
            self.axes["E"].set_title('$E (t)$')
        else:
            self.axes["E"].set_title('$(E-E_0) / E_0 (t)$')
        
        if self.diagnostics.plot_helicity:
            self.axes["H"].set_title('$H (t)$')
        else:
            self.axes["H"].set_title('$(H-H_0) / H_0 (t)$')
        
        self.axes["Emag" ].set_xlim((xStart,xEnd)) 
        self.axes["Evel" ].set_xlim((xStart,xEnd)) 
        self.axes["E"    ].set_xlim((xStart,xEnd)) 
        self.axes["H"    ].set_xlim((xStart,xEnd)) 
        
        self.axes["Emag" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["Evel" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["E"    ].yaxis.set_major_formatter(majorFormatter)
        self.axes["H"    ].yaxis.set_major_formatter(majorFormatter)
        
        
        # switch off some ticks
        plt.setp(self.axes["Bx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["By"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Vx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Vy"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["PB"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Emag" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Evel" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"    ].get_xticklabels(), visible=False)
        
        
        self.update()
        
    
    
    def read_data(self):
        
        self.B [0:-1, 0:-1] = self.diagnostics.B [:,:]
        self.B [  -1, 0:-1] = self.diagnostics.B [0,:]
        self.B [   :,   -1] = self.B[:,0]
        
        self.Bx[0:-1, 0:-1] = self.diagnostics.Bx[:,:]
        self.Bx[  -1, 0:-1] = self.diagnostics.Bx[0,:]
        self.Bx[   :,   -1] = self.Bx[:,0]
        
        self.By[0:-1, 0:-1] = self.diagnostics.By[:,:]
        self.By[  -1, 0:-1] = self.diagnostics.By[0,:]
        self.By[   :,   -1] = self.By[:,0]
        
        self.V [0:-1, 0:-1] = self.diagnostics.V [:,:]
        self.V [  -1, 0:-1] = self.diagnostics.V [0,:]
        self.V [   :,   -1] = self.V[:,0]
        
        self.Vx[0:-1, 0:-1] = self.diagnostics.Vx[:,:]
        self.Vx[  -1, 0:-1] = self.diagnostics.Vx[0,:]
        self.Vx[   :,   -1] = self.Vx[:,0]
        
        self.Vy[0:-1, 0:-1] = self.diagnostics.Vy[:,:]
        self.Vy[  -1, 0:-1] = self.diagnostics.Vy[0,:]
        self.Vy[   :,   -1] = self.Vy[:,0]
        
        self.P[0:-1, 0:-1] = self.diagnostics.P[:,:]
        self.P[  -1, 0:-1] = self.diagnostics.P[0,:]
        self.P[   :,   -1] = self.P[:,0]
        
        self.PB[0:-1, 0:-1] = self.diagnostics.e_magnetic[:,:]
        self.PB[  -1, 0:-1] = self.diagnostics.e_magnetic[0,:]
        self.PB[   :,   -1] = self.PB[:,0]
        
        self.J[0:-1, 0:-1] = self.diagnostics.J[:,:]
        self.J[  -1, 0:-1] = self.diagnostics.J[0,:]
        self.J[   :,   -1] = self.J[:,0]
        
        
    
    
    def update_boundaries(self):
        
        Bmin = min(self.diagnostics.Bx.min(), self.diagnostics.By.min(), -self.diagnostics.Bx.max(), -self.diagnostics.By.max())
        Bmax = max(self.diagnostics.Bx.max(), self.diagnostics.By.max(), -self.diagnostics.Bx.min(), -self.diagnostics.By.min())
        
        self.BxTicks = np.linspace(Bmin, Bmax, 11, endpoint=True)
        self.ByTicks = np.linspace(Bmin, Bmax, 11, endpoint=True)

#        self.BxNorm = colors.Normalize(vmin=Bxmin, vmax=Bxmax)
#        self.ByNorm = colors.Normalize(vmin=Bymin, vmax=Bymax)
        
        
        Vmin = 2. * min(self.diagnostics.Vx.min(), self.diagnostics.Vy.min(), -self.diagnostics.Vx.max(), -self.diagnostics.Vy.max())
        Vmax = 2. * max(self.diagnostics.Vx.max(), self.diagnostics.Vy.max(), -self.diagnostics.Vx.min(), -self.diagnostics.Vy.min())
        
        if Vmin == Vmax:
            Vmin -= 1.
            Vmax += 1.
        
        self.VxTicks = np.linspace(Vmin, Vmax, 11, endpoint=True)
        self.VyTicks = np.linspace(Vmin, Vmax, 11, endpoint=True)

        
        Pmin = min(self.diagnostics.P.min(), -self.diagnostics.P.max())
        Pmax = max(self.diagnostics.P.max(), -self.diagnostics.P.min())
        
        if Pmin == Pmax:
            Pmin -= 1.
            Pmax += 1.
        
        self.PTicks = np.linspace(Pmin, Pmax, 11, endpoint=True)
        
        
        PBmin = min(self.diagnostics.e_magnetic.min(), -self.diagnostics.e_magnetic.max())
        PBmax = min(self.diagnostics.e_magnetic.max(), -self.diagnostics.e_magnetic.min())
        PBdiff = (PBmax - PBmin)
        
        if PBmin == PBmax:
            PBmin -= 1.
            PBmax += 1.
        
#        self.PBnorm = colors.Normalize(vmin=PBmin, vmax=PBmax)
        self.PBnorm = colors.Normalize(vmin=PBmin - 0.2*PBdiff, vmax=PBmax + 0.2*PBdiff)
        self.PBTicks = np.linspace(PBmin - 0.2*PBdiff, PBmax + 0.2*PBdiff, 51, endpoint=True)
        
    
        Jmin = min(self.diagnostics.J.min(), -self.diagnostics.J.max())
        Jmax = min(self.diagnostics.J.max(), -self.diagnostics.J.min())
        Jdiff = (Jmax - Jmin)
        
        if Jmin == Jmax:
            Jmin -= 1.
            Jmax += 1.
        
#        self.Jnorm = colors.Normalize(vmin=Jmin, vmax=Jmax)
        self.Jnorm = colors.Normalize(vmin=Jmin - 0.2*Jdiff, vmax=Jmax + 0.2*Jdiff)
        self.JTicks = np.linspace(Jmin - 0.2*Jdiff, Jmax + 0.2*Jdiff, 51, endpoint=True)
        
    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0 or self.iTime-1 == self.nTime):
            return
        
        self.read_data()
#        self.update_boundaries()

        for ckey, cont in self.conts.items():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
        
        
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, self.BxTicks, cmap=cm.jet, extend='both')
#        self.cbars["Bx"] = self.figure.colorbar(self.conts["Bx"], ax=self.axes["Bx"], orientation='vertical', ticks=self.BxTicks)#, format='%0.2E'
       
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, self.ByTicks, cmap=cm.jet, extend='both')
#        self.cbars["By"] = self.figure.colorbar(self.conts["By"], ax=self.axes["By"], orientation='vertical', ticks=self.ByTicks)
        
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, self.VxTicks, cmap=cm.jet, extend='both')
#        self.cbars["Vx"] = self.figure.colorbar(self.conts["Vx"], ax=self.axes["Vx"], orientation='vertical', ticks=self.VxTicks)
        
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, self.VyTicks, cmap=cm.jet, extend='both')
#        self.cbars["Vy"] = self.figure.colorbar(self.conts["Vy"], ax=self.axes["Vy"], orientation='vertical', ticks=self.VyTicks)

#        self.conts["P"] = self.axes["P"].contourf(self.x, self.y, self.P.T, self.PTicks, cmap=cm.jet, extend='both')
        self.conts["P"] = self.axes["P"].contourf(self.x, self.y, self.P.T, self.PTicks, cmap=cm.jet, extend='both', ticks=self.PTicks)
        
#        self.conts["PB"] = self.axes["PB"].contourf(self.x, self.y, self.PB.T, ticks=self.PBTicks)#, extend='neither')
        self.conts["PB"] = self.axes["PB"].contourf(self.x, self.y, self.PB.T, 51, norm=self.PBnorm)#, extend='neither')
#        self.cbars["PB"] = self.figure.colorbar(self.conts["PB"], ax=self.axes["PB"], orientation='vertical', ticks=self.PBTicks)
        
#        self.conts["J"] = self.axes["J"].contourf(self.x, self.y, self.J.T, ticks=self.JTicks)#, extend='neither')
        self.conts["J"] = self.axes["J"].contourf(self.x, self.y, self.J.T, 51, norm=self.Jnorm)#, extend='neither')
#        self.cbars["J"] = self.figure.colorbar(self.conts["J"], ax=self.axes["J"], orientation='vertical', ticks=self.JTicks)
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()
        
        self.lines["Emag"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["Emag"].set_ydata(self.E_magnetic[tStart:tEnd])
        self.axes ["Emag"].relim()
        self.axes ["Emag"].autoscale_view()
        self.axes ["Emag"].set_xlim((xStart,xEnd)) 
        
        self.lines["Evel"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["Evel"].set_ydata(self.E_velocity[tStart:tEnd])
        self.axes ["Evel"].relim()
        self.axes ["Evel"].autoscale_view()
        self.axes ["Evel"].set_xlim((xStart,xEnd)) 
        
        self.lines["E"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["E"].set_ydata(self.energy[tStart:tEnd])
        self.axes ["E"].relim()
        self.axes ["E"].autoscale_view()
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        
        self.lines["H"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["H"].set_ydata(self.helicity[tStart:tEnd])
        self.axes ["H"].relim()
        self.axes ["H"].autoscale_view()
        self.axes ["H"].set_xlim((xStart,xEnd)) 
        
        
        plt.draw()
        plt.show(block=final)
        
        return self.figure
    
    
    def add_timepoint(self):
        
        self.E_magnetic [self.iTime] = self.diagnostics.E_magnetic
        self.E_velocity [self.iTime] = self.diagnostics.E_velocity
        
        if self.diagnostics.plot_energy:
            self.energy     [self.iTime] = self.diagnostics.energy
        else:
            self.energy     [self.iTime] = self.diagnostics.E_error
        
        if self.diagnostics.plot_helicity:
            self.helicity   [self.iTime] = self.diagnostics.helicity
        else:
            self.helicity   [self.iTime] = self.diagnostics.H_error
        
        
        self.title.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        
        self.iTime += 1
        
    
    def get_timerange(self):
        tStart = self.iTime - (self.nTime+1)
        tEnd   = self.iTime
        
        if tStart < 0:
            tStart = 0
        
        xStart = self.diagnostics.tGrid[tStart]
        xEnd   = self.diagnostics.tGrid[tStart+self.nTime]
        
        return tStart, tEnd, xStart, xEnd
    
