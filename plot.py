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
        
        
        self.L1_magnetic = np.zeros_like(diagnostics.tGrid)
        self.L1_velocity = np.zeros_like(diagnostics.tGrid)
        
        self.L2_magnetic = np.zeros_like(diagnostics.tGrid)
        self.L2_velocity = np.zeros_like(diagnostics.tGrid)
        
        self.E_velocity  = np.zeros_like(diagnostics.tGrid)
        self.E_magnetic  = np.zeros_like(diagnostics.tGrid)
        
        self.energy      = np.zeros_like(diagnostics.tGrid)
        self.helicity    = np.zeros_like(diagnostics.tGrid)
        
        
        self.x = np.zeros(diagnostics.nx+1)
        self.y = np.zeros(diagnostics.ny+1)
        
        self.x[0:-1] = self.diagnostics.xGrid
        self.x[  -1] = self.diagnostics.Lx
        
        self.y[0:-1] = self.diagnostics.yGrid
        self.y[  -1] = self.diagnostics.Ly
        
        self.Bx      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.By      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Vx      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Vy      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        self.B       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.V       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.2, wspace=0.25)
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
        
        self.update_boundaries()
        
        
        # create subplots
        gs = gridspec.GridSpec(4, 5)
        
        self.axes["Bx"]    = plt.subplot(gs[0,0])
        self.axes["By"]    = plt.subplot(gs[1,0])
        self.axes["Vx"]    = plt.subplot(gs[2,0])
        self.axes["Vy"]    = plt.subplot(gs[3,0])
        self.axes["Babs"]  = plt.subplot(gs[0:2,1:3])
        self.axes["Vabs"]  = plt.subplot(gs[2:4,1:3])
        self.axes["Emag"]  = plt.subplot(gs[0,3])
        self.axes["Evel"]  = plt.subplot(gs[1,3])
        self.axes["E"]     = plt.subplot(gs[2,3])
        self.axes["H"]     = plt.subplot(gs[3,3])
        self.axes["L1mag"] = plt.subplot(gs[0,4])
        self.axes["L1vel"] = plt.subplot(gs[1,4])
        self.axes["L2mag"] = plt.subplot(gs[2,4])
        self.axes["L2vel"] = plt.subplot(gs[3,4])
        
        
        self.axes["Bx"].set_title('$B_{x} (x,y)$')
        self.axes["By"].set_title('$B_{y} (x,y)$')
        self.axes["Vx"].set_title('$V_{x} (x,y)$')
        self.axes["Vy"].set_title('$V_{y} (x,y)$')
        self.axes["Babs"].set_title('$B (x,y)$')
        self.axes["Vabs"].set_title('$V (x,y)$')
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()

        self.lines["Emag" ], = self.axes["Emag" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.E_magnetic [tStart:tEnd])
        self.lines["Evel" ], = self.axes["Evel" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.E_velocity [tStart:tEnd])
        self.lines["E"    ], = self.axes["E"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.energy     [tStart:tEnd])
        self.lines["H"    ], = self.axes["H"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.helicity   [tStart:tEnd])
        
        self.lines["L1mag"], = self.axes["L1mag"].plot(self.diagnostics.tGrid[tStart:tEnd], self.L1_magnetic[tStart:tEnd])
        self.lines["L1vel"], = self.axes["L1vel"].plot(self.diagnostics.tGrid[tStart:tEnd], self.L1_velocity[tStart:tEnd])
        self.lines["L2mag"], = self.axes["L2mag"].plot(self.diagnostics.tGrid[tStart:tEnd], self.L2_magnetic[tStart:tEnd])
        self.lines["L2vel"], = self.axes["L2vel"].plot(self.diagnostics.tGrid[tStart:tEnd], self.L2_velocity[tStart:tEnd])
        
        self.axes["Emag"].set_title('$E_{B} (t)$')
        self.axes["Evel"].set_title('$E_{V} (t)$')
        self.axes["E"].set_title('$\Delta E (t)$')
        self.axes["H"].set_title('$\Delta H (t)$')
        
        self.axes["L1mag"].set_title('$\Delta L_{1}^{B} (t)$')
        self.axes["L1vel"].set_title('$\Delta L_{1}^{V} (t)$')
        self.axes["L2mag"].set_title('$\Delta L_{2}^{B} (t)$')
        self.axes["L2vel"].set_title('$\Delta L_{2}^{V} (t)$')

        self.axes["Emag" ].set_xlim((xStart,xEnd)) 
        self.axes["Evel" ].set_xlim((xStart,xEnd)) 
        self.axes["E"    ].set_xlim((xStart,xEnd)) 
        self.axes["H"    ].set_xlim((xStart,xEnd)) 
        
        self.axes["L1mag"].set_xlim((xStart,xEnd)) 
        self.axes["L1vel"].set_xlim((xStart,xEnd)) 
        self.axes["L2mag"].set_xlim((xStart,xEnd)) 
        self.axes["L2vel"].set_xlim((xStart,xEnd)) 
        
        self.axes["Emag" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["Evel" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["E"    ].yaxis.set_major_formatter(majorFormatter)
        self.axes["H"    ].yaxis.set_major_formatter(majorFormatter)
        
        self.axes["L1mag"].yaxis.set_major_formatter(majorFormatter)
        self.axes["L1vel"].yaxis.set_major_formatter(majorFormatter)
        self.axes["L2mag"].yaxis.set_major_formatter(majorFormatter)
        self.axes["L2vel"].yaxis.set_major_formatter(majorFormatter)
        
        
        # switch off some ticks
        plt.setp(self.axes["Bx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["By"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Vx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Babs" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Emag" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Evel" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"    ].get_xticklabels(), visible=False)
        plt.setp(self.axes["L1mag"].get_xticklabels(), visible=False)
        plt.setp(self.axes["L1vel"].get_xticklabels(), visible=False)
        plt.setp(self.axes["L2mag"].get_xticklabels(), visible=False)
        
        
        self.update()
        
    
    def update_boundaries(self):
        self.Bmin = +1e40
        self.Bmax = -1e40
        
        self.Bmin = min(self.Bmin, self.diagnostics.B.min() )
        self.Bmin = min(self.Bmin, self.diagnostics.Bx.min() )
        self.Bmin = min(self.Bmin, self.diagnostics.By.min() )
        
        self.Bmax = max(self.Bmax, self.diagnostics.B.max() )
        self.Bmax = max(self.Bmax, self.diagnostics.Bx.max() )
        self.Bmax = max(self.Bmax, self.diagnostics.By.max() )

        dB = 0.1 * (self.Bmax - self.Bmin)
        self.Bnorm = colors.Normalize(vmin=self.Bmin-dB, vmax=self.Bmax+dB)


        self.Vmin = +1e40
        self.Vmax = -1e40
        
        self.Vmin = min(self.Vmin, self.diagnostics.V.min() )
        self.Vmin = min(self.Vmin, self.diagnostics.Vx.min() )
        self.Vmin = min(self.Vmin, self.diagnostics.Vy.min() )
        
        self.Vmax = max(self.Vmax, self.diagnostics.V.max() )
        self.Vmax = max(self.Vmax, self.diagnostics.Vx.max() )
        self.Vmax = max(self.Vmax, self.diagnostics.Vy.max() )

        dV = 0.1 * (self.Vmax - self.Vmin)
        self.Vnorm = colors.Normalize(vmin=self.Vmin-dV, vmax=self.Vmax+dV)
        
        
    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0 or self.iTime-1 == self.nTime):
            return
        
#        self.update_boundaries()

        for ckey, cont in self.conts.iteritems():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
        
        
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
        
        
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx, 20, norm=self.Bnorm)
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By, 20, norm=self.Bnorm)
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx, 20, norm=self.Vnorm)
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy, 20, norm=self.Vnorm)

        self.conts["Babs"] = self.axes["Babs"].contourf(self.x, self.y, self.B, 20, norm=self.Bnorm)
        self.conts["Vabs"] = self.axes["Vabs"].contourf(self.x, self.y, self.V, 20, norm=self.Vnorm)
        
        
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
        
        self.lines["L1mag"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["L1mag"].set_ydata(self.L1_magnetic[tStart:tEnd])
        self.axes ["L1mag"].relim()
        self.axes ["L1mag"].autoscale_view()
        self.axes ["L1mag"].set_xlim((xStart,xEnd)) 
        
        self.lines["L1vel"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["L1vel"].set_ydata(self.L1_velocity[tStart:tEnd])
        self.axes ["L1vel"].relim()
        self.axes ["L1vel"].autoscale_view()
        self.axes ["L1vel"].set_xlim((xStart,xEnd)) 
        
        self.lines["L2mag"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["L2mag"].set_ydata(self.L2_magnetic[tStart:tEnd])
        self.axes ["L2mag"].relim()
        self.axes ["L2mag"].autoscale_view()
        self.axes ["L2mag"].set_xlim((xStart,xEnd)) 
        
        self.lines["L2vel"].set_xdata(self.diagnostics.tGrid[tStart:tEnd])
        self.lines["L2vel"].set_ydata(self.L2_velocity[tStart:tEnd])
        self.axes ["L2vel"].relim()
        self.axes ["L2vel"].autoscale_view()
        self.axes ["L2vel"].set_xlim((xStart,xEnd)) 
        
        
        plt.draw()
        plt.show(block=final)
        
        return self.figure
    
    
    def add_timepoint(self):
        
        self.E_magnetic [self.iTime] = self.diagnostics.E_magnetic
        self.E_velocity [self.iTime] = self.diagnostics.E_velocity
        self.energy     [self.iTime] = self.diagnostics.E_error
        self.helicity   [self.iTime] = self.diagnostics.H_error
        self.L1_magnetic[self.iTime] = self.diagnostics.L1_magnetic_error
        self.L1_velocity[self.iTime] = self.diagnostics.L1_velocity_error
        self.L2_magnetic[self.iTime] = self.diagnostics.L2_magnetic_error
        self.L2_velocity[self.iTime] = self.diagnostics.L2_velocity_error
        
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
    
