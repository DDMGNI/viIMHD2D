'''
Created on Nov 26, 2015

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter

from .plot_base import PlotMHD2Dbase


class PlotMHD2DbaseReplay(PlotMHD2Dbase):
    '''
    Basic plotting class for replay plots.
    '''

    def __init__(self, diagnostics, nTime=0, nPlot=1):
        '''
        Constructor.
        '''
        
        super().__init__(diagnostics, nTime=0, nPlot=1)
        
        self.nrows = 2
        self.ncols = 4
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=1.75, wspace=0.25)
        plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = 0.0' % (diagnostics.tGrid[0]), horizontalalignment='center') 
        
        # set up tick formatter
        self.majorFormatter = ScalarFormatter(useOffset=True)
        self.majorFormatter.set_powerlimits((-2,+2))
        self.majorFormatter.set_scientific(True)

#         majorFormatter = FormatStrFormatter('%2.2e')

        majorFormatter_errors = ScalarFormatter(useOffset=False)
        majorFormatter_errors.set_powerlimits((-1,+1))
        majorFormatter_errors.set_scientific(True)

        # set up plots
        self.axes  = {}
        self.conts = {}
        self.cbars = {}
        self.lines = {}
        self.vecs  = {}
        
        
        # create subplots
        gs = gridspec.GridSpec(20, 5)
        self.gs = gs
        
        self.axes["Bx"]   = plt.subplot(gs[ 0: 4,  0])
        self.axes["By"]   = plt.subplot(gs[ 4: 8,  0])
        self.axes["Vx"]   = plt.subplot(gs[ 8:12,  0])
        self.axes["Vy"]   = plt.subplot(gs[12:16,  0])
        self.axes["P"]    = plt.subplot(gs[16:20,  0])
        self.axes["M1"]   = plt.subplot(gs[ 0:10,1:3])
        self.axes["M2"]   = plt.subplot(gs[10:20,1:3])
        self.axes["Emag"] = plt.subplot(gs[ 0: 5,3:5])
        self.axes["Evel"] = plt.subplot(gs[ 5:10,3:5])
        self.axes["E"]    = plt.subplot(gs[10:15,3:5])
        self.axes["H"]    = plt.subplot(gs[15:20,3:5])
        
        self.axes["Bx"].set_title('$B_{x} (x,y)$')
        self.axes["By"].set_title('$B_{y} (x,y)$')
        self.axes["Vx"].set_title('$V_{x} (x,y)$')
        self.axes["Vy"].set_title('$V_{y} (x,y)$')
        self.axes["P" ].set_title('$P (x,y)$')
        
        self.lines["Emag" ], = self.axes["Emag" ].plot(self.diagnostics.tGrid[0], self.E_magnetic [0])
        self.lines["Evel" ], = self.axes["Evel" ].plot(self.diagnostics.tGrid[0], self.E_velocity [0])
        self.lines["E"    ], = self.axes["E"    ].plot(self.diagnostics.tGrid[0], self.energy     [0])
        self.lines["H"    ], = self.axes["H"    ].plot(self.diagnostics.tGrid[0], self.helicity   [0])
        
        self.axes["Emag"].set_title('$E_{B} (t)$')
        self.axes["Evel"].set_title('$E_{V} (t)$')
        
        self.axes["Emag" ].yaxis.set_major_formatter(majorFormatter_errors)
        self.axes["Evel" ].yaxis.set_major_formatter(majorFormatter_errors)
        
        if self.diagnostics.plot_energy:
            self.axes["E"].set_title('$E (t)$')
            self.axes["E"].yaxis.set_major_formatter(majorFormatter_errors)
        else:
            self.axes["E"].set_title('$(E (t) - E (0)) / E (0)$')
            self.axes["E"].yaxis.set_major_formatter(majorFormatter_errors)
        
        if self.diagnostics.plot_helicity:
            self.axes["H"].set_title('$H (t)$')
            self.axes["H"].yaxis.set_major_formatter(majorFormatter_errors)
        else:
            self.axes["H"].set_title('$(H (t) - H (0)) / H (0)$')
            self.axes["H"].yaxis.set_major_formatter(majorFormatter_errors)
        
        
        # switch off some ticks
        plt.setp(self.axes["Bx"  ].get_xticklabels(), visible=False)
        plt.setp(self.axes["By"  ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Vx"  ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Vy"  ].get_xticklabels(), visible=False)
        plt.setp(self.axes["M1"  ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Emag"].get_xticklabels(), visible=False)
        plt.setp(self.axes["Evel"].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"   ].get_xticklabels(), visible=False)
        
        
        self.update(0)
        
    
    def update(self, iTime, final=False, draw=True):
        '''
        Update plot.
        '''
        
        self.E_magnetic[iTime] = self.diagnostics.E_magnetic
        self.E_velocity[iTime] = self.diagnostics.E_velocity
        
        if self.diagnostics.plot_energy:
            self.energy     [iTime] = self.diagnostics.energy
        else:
            self.energy     [iTime] = self.diagnostics.E_error
        
        if self.diagnostics.plot_helicity:
            self.helicity   [iTime] = self.diagnostics.helicity
        else:
            self.helicity   [iTime] = self.diagnostics.H_error
        
        
        self.title.set_text('t = %1.2f' % (self.diagnostics.tGrid[iTime]))
        
        
        if not (iTime == 0 or (iTime-1) % self.nPlot == 0 or iTime-1 == self.nTime):
            return
        
        
        self.load_data()
        self.update_boundaries_pressure()
        
        
        for ckey, cont in self.conts.items():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
        
        
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, self.BxTicks, cmap=cm.viridis, extend='both')
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, self.ByTicks, cmap=cm.viridis, extend='both')
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, self.VxTicks, cmap=cm.viridis, extend='both')
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, self.VyTicks, cmap=cm.viridis, extend='both')
        self.conts["P" ] = self.axes["P" ].contourf(self.x, self.y, self.P.T,  self.PTicks,  cmap=cm.viridis, extend='both', ticks=self.PTicks)
        
        if iTime == 0:
            self.cbars["Bx"] = self.figure.colorbar(self.conts["Bx"], ax=self.axes["Bx"], orientation='vertical', format=self.majorFormatter, ticks=self.BxTicks)
            self.cbars["By"] = self.figure.colorbar(self.conts["By"], ax=self.axes["By"], orientation='vertical', format=self.majorFormatter, ticks=self.ByTicks)
            self.cbars["Vx"] = self.figure.colorbar(self.conts["Vx"], ax=self.axes["Vx"], orientation='vertical', format=self.majorFormatter, ticks=self.VxTicks)
            self.cbars["Vy"] = self.figure.colorbar(self.conts["Vy"], ax=self.axes["Vy"], orientation='vertical', format=self.majorFormatter, ticks=self.VyTicks)
            self.cbars["P" ] = self.figure.colorbar(self.conts["P" ], ax=self.axes["P" ], orientation='vertical', format=self.majorFormatter, ticks=self.PTicks)
        
        
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
        
        
        if draw:
            self.figure.canvas.draw()
#             self.figure.show()
#             plt.draw()
            plt.show(block=final)
            plt.pause(1)
        
        return self.figure
