'''
Created on Jul 02, 2012

@author: mkraus
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib import rc, rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter, FuncFormatter


def scinot(x,pos=None):
    if x == 0:
        s = '0'
    else:
        xp = int(np.floor(np.log10(np.abs(x))))
        mn = x/10.**xp
        # Here we truncate to 2 significant digits -- may not be enough
        # in all cases
        s = '$'+str('%.3f'%mn) +'\\times 10^{'+str(xp)+'}$'
        return s


class PlotMHD2D(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, nTime=0, ntMax=0, nPlot=1, write=False):
        '''
        Constructor
        '''
        
#        rc('text.latex',preamble=r'\usepackage[math]{helvet}')
#        rcParams.update({'font.sans-serif': 'Helvetica'})
#        rc('text',usetex=True)
        
        self.write = write
        self.prefix = '_petscMHD2D_'
        
        self.nrows = 2
        self.ncols = 4
        
        if ntMax == 0:
            ntMax = diagnostics.nt
        
        if nTime > 0 and nTime < ntMax:
            self.nTime = nTime
        else:
            self.nTime = ntMax
        
        self.iTime = 0
        self.nPlot = nPlot
        
        self.diagnostics = diagnostics
        
        
        self.E_velocity0 = self.diagnostics.E_velocity
        self.E_magnetic0 = self.diagnostics.E_magnetic
        
        self.E_velocity  = np.zeros(ntMax+1)
        self.E_magnetic  = np.zeros(ntMax+1)
        
        self.energy      = np.zeros(ntMax+1)
        self.helicity    = np.zeros(ntMax+1)
        
        
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
        
        self.A       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.J       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.PB      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
                
        self.B       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.V       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.25, wspace=0.2)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)
        
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
        gs = gridspec.GridSpec(4, 3)
        self.gs = gs
        
        self.axes["Bx"]    = plt.subplot(gs[0,0])
        self.axes["By"]    = plt.subplot(gs[0,1])
        self.axes["Vx"]    = plt.subplot(gs[1,0])
        self.axes["Vy"]    = plt.subplot(gs[1,1])
        self.axes["P"]     = plt.subplot(gs[2:4,0:2])
        self.axes["Emag"]  = plt.subplot(gs[0,2])
        self.axes["Evel"]  = plt.subplot(gs[1,2])
        self.axes["E"]     = plt.subplot(gs[2,2])
        self.axes["H"]     = plt.subplot(gs[3,2])
        
        
        self.axes["Bx"].set_title('$B_{x} (x,y)$')
        self.axes["By"].set_title('$B_{y} (x,y)$')
        self.axes["Vx"].set_title('$V_{x} (x,y)$')
        self.axes["Vy"].set_title('$V_{y} (x,y)$')
        self.axes["P" ].set_title('$A (x,y)$')
        
        
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, self.BxTicks, cmap=cm.jet, extend='both')
        self.cbars["Bx"] = self.figure.colorbar(self.conts["Bx"], ax=self.axes["Bx"], orientation='vertical', ticks=self.BxTicks)
        self.cbars["Bx"].ax.set_yticklabels(self.BxTickLabels)
       
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, self.ByTicks, cmap=cm.jet, extend='both')
        self.cbars["By"] = self.figure.colorbar(self.conts["By"], ax=self.axes["By"], orientation='vertical', ticks=self.ByTicks)
        self.cbars["By"].ax.set_yticklabels(self.ByTickLabels)
        
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, self.VxTicks, cmap=cm.jet, extend='both')
        self.cbars["Vx"] = self.figure.colorbar(self.conts["Vx"], ax=self.axes["Vx"], orientation='vertical', ticks=self.VxTicks)
        
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, self.VyTicks, cmap=cm.jet, extend='both')
        self.cbars["Vy"] = self.figure.colorbar(self.conts["Vy"], ax=self.axes["Vy"], orientation='vertical', ticks=self.VyTicks)
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()

        self.lines["Emag" ], = self.axes["Emag" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.E_magnetic [tStart:tEnd])
        self.lines["Evel" ], = self.axes["Evel" ].plot(self.diagnostics.tGrid[tStart:tEnd], self.E_velocity [tStart:tEnd])
        self.lines["E"    ], = self.axes["E"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.energy     [tStart:tEnd])
        self.lines["H"    ], = self.axes["H"    ].plot(self.diagnostics.tGrid[tStart:tEnd], self.helicity   [tStart:tEnd])
        
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
        
        self.axes["Emag" ].set_xlim((xStart,xEnd)) 
        self.axes["Evel" ].set_xlim((xStart,xEnd)) 
        self.axes["E"    ].set_xlim((xStart,xEnd)) 
        self.axes["H"    ].set_xlim((xStart,xEnd)) 
        
        self.axes["Emag" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["Evel" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["E"    ].yaxis.set_major_formatter(majorFormatter)
        self.axes["H"    ].yaxis.set_major_formatter(majorFormatter)
        
#        formatter = FuncFormatter(scinot)
#        
#        self.axes["Emag" ].yaxis.set_major_formatter(formatter)
#        self.axes["Evel" ].yaxis.set_major_formatter(formatter)
#        self.axes["E"    ].yaxis.set_major_formatter(formatter)
#        self.axes["H"    ].yaxis.set_major_formatter(formatter)
        
        
        # switch off some ticks
        plt.setp(self.axes["Bx"   ].get_xticklabels(), visible=False)
        plt.setp(self.axes["By"   ].get_xticklabels(), visible=False)
#        plt.setp(self.axes["Vx"   ].get_xticklabels(), visible=False)
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
        
        self.A[0:-1, 0:-1] = self.diagnostics.A[:,:]
        self.A[  -1, 0:-1] = self.diagnostics.A[0,:]
        self.A[   :,   -1] = self.A[:,0]
        
        self.J[0:-1, 0:-1] = self.diagnostics.J[:,:]
        self.J[  -1, 0:-1] = self.diagnostics.J[0,:]
        self.J[   :,   -1] = self.J[:,0]
        
        self.PB[0:-1, 0:-1] = self.diagnostics.e_magnetic[:,:]
        self.PB[  -1, 0:-1] = self.diagnostics.e_magnetic[0,:]
        self.PB[   :,   -1] = self.PB[:,0]
        
    
    
    def update_boundaries(self):
        
        Bmin = min(self.diagnostics.Bx.min(), self.diagnostics.By.min(), -self.diagnostics.Bx.max(), -self.diagnostics.By.max())
        Bmax = max(self.diagnostics.Bx.max(), self.diagnostics.By.max(), -self.diagnostics.Bx.min(), -self.diagnostics.By.min())
        
#        if Bmin == Bmax:
#            Bmin -= .1 * Bmin
#            Bmax += .1 * Bmax
        
        self.BxTicks = np.linspace(Bmin, Bmax, 11, endpoint=True)
        self.ByTicks = np.linspace(Bmin, Bmax, 11, endpoint=True)
        
        self.BxTickLabels = ["$%s \\times 10^{-3}$" % ( "%+1.1f" % (t * 1.E3)) for t in self.BxTicks]
        self.ByTickLabels = ["$%s \\times 10^{-3}$" % ( "%+1.1f" % (t * 1.E3)) for t in self.ByTicks]
        
        for i in range(0, len(self.BxTickLabels)):
            if (i+1) % 2 == 0:
                self.BxTickLabels[i] = ''

        for i in range(0, len(self.ByTickLabels)):
            if (i+1) % 2 == 0:
                self.ByTickLabels[i] = ''

        
        Vxmin = self.diagnostics.Vx.min()
        Vxmax = self.diagnostics.Vx.max()
        
        if Vxmin == Vxmax:
            Vxmin = np.round(Vxmin, 2) - 0.01
            Vxmax = np.round(Vxmax, 2) + 0.01
        
        Vymin = self.diagnostics.Vy.min()
        Vymax = self.diagnostics.Vy.max()
        
        if Vymin == Vymax:
            Vymin = np.round(Vymin, 2) - 0.01
            Vymax = np.round(Vymax, 2) + 0.01
        
        self.VxTicks = np.linspace(Vxmin, Vxmax, 5, endpoint=True)
        self.VyTicks = np.linspace(Vymin, Vymax, 5, endpoint=True)
        
        
        PBmin = min(self.diagnostics.e_magnetic.min(), -self.diagnostics.e_magnetic.max())
        PBmax = min(self.diagnostics.e_magnetic.max(), -self.diagnostics.e_magnetic.min())
        PBdiff = (PBmax - PBmin)
        
        if PBmin == PBmax:
            PBmin -= .1 * PBmin
            PBmax += .1 * PBmax
        
        self.PBnorm = colors.Normalize(vmin=PBmin - 0.2*PBdiff, vmax=PBmax + 0.2*PBdiff)
        self.PBTicks = np.linspace(PBmin - 0.2*PBdiff, PBmax + 0.2*PBdiff, 51, endpoint=True)


        Amin = min(self.diagnostics.A.min(), -self.diagnostics.A.max())
        Amax = max(self.diagnostics.A.max(), -self.diagnostics.A.min())
        Adif = Amax - Amin
        
        self.ATicks = np.linspace(Amin + 0.45 * Adif, Amax, 10)

    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0 or self.iTime-1 == self.nTime):
            return
        
        self.read_data()
#        self.update_boundaries()

        for ckey, cont in self.conts.items():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
        
        
        self.Bx[0:-1, 0:-1] = self.diagnostics.Bx[:,:]
        self.Bx[  -1, 0:-1] = self.diagnostics.Bx[0,:]
        self.Bx[   :,   -1] = self.Bx[:,0]
        
        self.By[0:-1, 0:-1] = self.diagnostics.By[:,:]
        self.By[  -1, 0:-1] = self.diagnostics.By[0,:]
        self.By[   :,   -1] = self.By[:,0]
        
        self.Vx[0:-1, 0:-1] = self.diagnostics.Vx[:,:]
        self.Vx[  -1, 0:-1] = self.diagnostics.Vx[0,:]
        self.Vx[   :,   -1] = self.Vx[:,0]
        
        self.Vy[0:-1, 0:-1] = self.diagnostics.Vy[:,:]
        self.Vy[  -1, 0:-1] = self.diagnostics.Vy[0,:]
        self.Vy[   :,   -1] = self.Vy[:,0]
        
        
        
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T)
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T)
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T)
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T)
        
#        self.conts["P"] = self.axes["P"].contourf(self.x, self.y, self.PB.T, 51, norm=self.PBnorm)
        self.conts["P"] = self.axes["P"].contour(self.x, self.y, self.A.T, levels=self.ATicks, cmap=cm.jet, extend='neither')
        
        
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
        
        
        if self.write:
            filename = self.prefix + str('%06d' % self.iTime) + '.png'
            plt.savefig(filename, dpi=100)
#            filename = self.prefix + str('%06d' % self.iTime) + '.pdf'
#            plt.savefig(filename)
        else:
            plt.draw()
            plt.show(block=final)
    
    
    def add_timepoint(self):
        
        self.E_magnetic [self.iTime] = self.diagnostics.E_magnetic - self.E_magnetic0
        self.E_velocity [self.iTime] = self.diagnostics.E_velocity - self.E_velocity0
        
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
    
