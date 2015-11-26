'''
Created on 26.11.2015

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter

from .plot_base import PlotMHD2Dbase

class PlotMHD2DbaseMovie(PlotMHD2Dbase):
    '''
    Basic plotting class for movie_reconnection plots.
    '''

    def __init__(self, diagnostics, nTime=0, nPlot=1, ntMax=0, write=False, prefix='viIMHD2D_'):
        '''
        Constructor.
        '''
        
        super().__init__(diagnostics, nTime=nTime, nPlot=nPlot, ntMax=ntMax, write=write, prefix=prefix)
        
        self.nrows = 2
        self.ncols = 4
        
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.25, wspace=0.2)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = 0.0' % (diagnostics.tGrid[0]), horizontalalignment='center') 
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        ## -> limit to 1.1f precision
        majorFormatter.set_powerlimits((-1,+1))
        majorFormatter.set_scientific(True)

        # set up plots
        self.axes  = {}
        self.conts = {}
        self.cbars = {}
        self.lines = {}
        self.vecs  = {}
        
        self.update_boundaries()
        
        
        # create subplots
        gs = gridspec.GridSpec(4, 3)
        self.gs = gs
        
        self.axes["Bx"]    = plt.subplot(gs[0,0])
        self.axes["By"]    = plt.subplot(gs[0,1])
        self.axes["Vx"]    = plt.subplot(gs[1,0])
        self.axes["Vy"]    = plt.subplot(gs[1,1])
        self.axes["M"]     = plt.subplot(gs[2:4,0:2])
        self.axes["Emag"]  = plt.subplot(gs[0,2])
        self.axes["Evel"]  = plt.subplot(gs[1,2])
        self.axes["E"]     = plt.subplot(gs[2,2])
        self.axes["H"]     = plt.subplot(gs[3,2])
        
        
        self.axes["Bx"].set_title('$B_{x} (x,y)$')
        self.axes["By"].set_title('$B_{y} (x,y)$')
        self.axes["Vx"].set_title('$V_{x} (x,y)$')
        self.axes["Vy"].set_title('$V_{y} (x,y)$')
        
        
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, self.BxTicks, cmap=cm.viridis, extend='both')
        self.cbars["Bx"] = self.figure.colorbar(self.conts["Bx"], ax=self.axes["Bx"], orientation='vertical', ticks=self.BxTicks)
       
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, self.ByTicks, cmap=cm.viridis, extend='both')
        self.cbars["By"] = self.figure.colorbar(self.conts["By"], ax=self.axes["By"], orientation='vertical', ticks=self.ByTicks)
        
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, self.VxTicks, cmap=cm.viridis, extend='both')
        self.cbars["Vx"] = self.figure.colorbar(self.conts["Vx"], ax=self.axes["Vx"], orientation='vertical', ticks=self.VxTicks)
        
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, self.VyTicks, cmap=cm.viridis, extend='both')
        self.cbars["Vy"] = self.figure.colorbar(self.conts["Vy"], ax=self.axes["Vy"], orientation='vertical', ticks=self.VyTicks)
        
        
        self.lines["Emag" ], = self.axes["Emag" ].plot(self.diagnostics.tGrid[0], self.E_magnetic [0])
        self.lines["Evel" ], = self.axes["Evel" ].plot(self.diagnostics.tGrid[0], self.E_velocity [0])
        self.lines["E"    ], = self.axes["E"    ].plot(self.diagnostics.tGrid[0], self.energy     [0])
        self.lines["H"    ], = self.axes["H"    ].plot(self.diagnostics.tGrid[0], self.helicity   [0])
        
        self.axes["Emag"].set_title('$E_{B} (t) - E_{B} (0)$')
        self.axes["Evel"].set_title('$E_{V} (t) - E_{V} (0)$')
        
        if self.diagnostics.plot_energy:
            self.axes["E"].set_title('$E (t)$')
        else:
            self.axes["E"].set_title('$(E-E_0) / E_0 (t)$')
        
        if self.diagnostics.plot_helicity:
            self.axes["H"].set_title('$H (t)$')
        else:
            self.axes["H"].set_title('$(H-H_0) / H_0 (t)$')
        
        self.axes["Emag" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["Evel" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["E"    ].yaxis.set_major_formatter(majorFormatter)
        self.axes["H"    ].yaxis.set_major_formatter(majorFormatter)
        
        
        # switch off some ticks
        plt.setp(self.axes["Bx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["By"   ].get_xticklabels(), visible=False)
#        plt.setp(self.axes["Vx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Emag" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Evel" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"    ].get_xticklabels(), visible=False)
        
        
#         self.update(0)
        
        
    
    def update_sub(self, iTime):
        '''
        Update plot.
        '''
        
        self.update_boundaries_pressure()
        
        
        for ckey, cont in self.conts.items():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
        
        
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, self.BxTicks, cmap=cm.viridis, extend='both')
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, self.ByTicks, cmap=cm.viridis, extend='both')
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, self.VxTicks, cmap=cm.viridis, extend='both')
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, self.VyTicks, cmap=cm.viridis, extend='both')
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange(iTime)
        
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
        

