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
        
#        plt.ion()
        
        self.nrows = 2
        self.ncols = 4
        
        if nTime > 0 and nTime < diagnostics.nt:
            self.nTime = nTime
        else:
            self.nTime = diagnostics.nt
        
        self.iTime = 0
        self.nPlot = nPlot
        
        self.diagnostics = diagnostics
        
        
#        self.L1_magnetic = np.zeros_like(diagnostics.tGrid)
#        self.L1_velocity = np.zeros_like(diagnostics.tGrid)
#        
#        self.L2_magnetic = np.zeros_like(diagnostics.tGrid)
#        self.L2_velocity = np.zeros_like(diagnostics.tGrid)
        
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
        
        self.A       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.J       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.PB      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
                
        self.B       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.V       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        self.divB    = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.divV    = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=1.75, wspace=0.25)
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
        
        self.update_boundaries()
        
        
        # create subplots
        gs = gridspec.GridSpec(20, 5)
        self.gs = gs
        
        self.axes["Bx"]    = plt.subplot(gs[ 0: 4,  0])
        self.axes["By"]    = plt.subplot(gs[ 4: 8,  0])
        self.axes["Vx"]    = plt.subplot(gs[ 8:12,  0])
        self.axes["Vy"]    = plt.subplot(gs[12:16,  0])
        self.axes["P"]     = plt.subplot(gs[16:20,  0])
        self.axes["Babs"]  = plt.subplot(gs[ 0:10,1:3])
        self.axes["Vabs"]  = plt.subplot(gs[10:20,1:3])
        self.axes["Emag"]  = plt.subplot(gs[ 0: 5,3:5])
        self.axes["Evel"]  = plt.subplot(gs[ 5:10,3:5])
        self.axes["E"]     = plt.subplot(gs[10:15,3:5])
        self.axes["H"]     = plt.subplot(gs[15:20,3:5])
        
        
        self.axes["Bx"].set_title('$B_{x} (x,y)$')
        self.axes["By"].set_title('$B_{y} (x,y)$')
        self.axes["Vx"].set_title('$V_{x} (x,y)$')
        self.axes["Vy"].set_title('$V_{y} (x,y)$')
        self.axes["P" ].set_title('$P (x,y)$')
        self.axes["Babs"].set_title('$div \, V  (x,y)$')
        self.axes["Vabs"].set_title('$B (x,y)$')
        
        
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, self.BxTicks, cmap=cm.jet, extend='both')
        self.cbars["Bx"] = self.figure.colorbar(self.conts["Bx"], ax=self.axes["Bx"], orientation='vertical', ticks=self.BxTicks)#, format='%0.2E'
       
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, self.ByTicks, cmap=cm.jet, extend='both')
        self.cbars["By"] = self.figure.colorbar(self.conts["By"], ax=self.axes["By"], orientation='vertical', ticks=self.ByTicks)
        
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, self.VxTicks, cmap=cm.jet, extend='both')
        self.cbars["Vx"] = self.figure.colorbar(self.conts["Vx"], ax=self.axes["Vx"], orientation='vertical', ticks=self.VxTicks)
        
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, self.VyTicks, cmap=cm.jet, extend='both')
        self.cbars["Vy"] = self.figure.colorbar(self.conts["Vy"], ax=self.axes["Vy"], orientation='vertical', ticks=self.VyTicks)
        
        self.conts["P" ] = self.axes["P"].contourf(self.x, self.y, self.P.T, self.PTicks, cmap=cm.jet, extend='both')
        self.cbars["P" ] = self.figure.colorbar(self.conts["P"], ax=self.axes["P"], orientation='vertical', ticks=self.PTicks)
        
        self.conts["Vabs"] = self.axes["Vabs"].contour(self.x, self.y, self.A.T, self.ATicks, cmap=cm.jet, extend='neither')
        
#        self.conts["Babs"] = self.axes["Babs"].contourf(self.x, self.y, self.divV.T, self.divVTicks, cmap=cm.jet)
#        self.cbars["Babs"] = self.figure.colorbar(self.conts["Babs"], ax=self.axes["Babs"], orientation='vertical', ticks=self.divVTicks)

#        plt.subplot(self.gs[0:10,1:3])
#        plt.streamplot(self.x, self.y, self.Bx.T, self.By.T, density=1.2, arrowstyle='-', arrowsize=.01, minlength=.2, color='b')
        
        
#        self.stream_xstart = np.linspace(0., 2., 25)
#        self.stream_ystart = np.zeros(25)
#        self.stream_ystart[7:19] = 2.
        
        
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
        plt.setp(self.axes["Babs" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Emag" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["Evel" ].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"    ].get_xticklabels(), visible=False)
        
        
        self.update()
        
    
    def update_boundaries(self):
        
        Bmin = min(self.diagnostics.Bx.min(), self.diagnostics.By.min(), -self.diagnostics.Bx.max(), -self.diagnostics.By.max())
        Bmax = max(self.diagnostics.Bx.max(), self.diagnostics.By.max(), -self.diagnostics.Bx.min(), -self.diagnostics.By.min())
        
#        Bxmin = self.diagnostics.Bx.min()
#        Bxmax = self.diagnostics.Bx.max()
#        
#        Bymin = self.diagnostics.By.min()
#        Bymax = self.diagnostics.By.max()
#
#        if Bxmin == Bxmax:
#            Bxmin -= 1.
#            Bxmax += 1.
#        
#        if Bymin == Bymax:
#            Bymin -= 1.
#            Bymax += 1.
        
        self.BxTicks = np.linspace(Bmin, Bmax, 11, endpoint=True)
        self.ByTicks = np.linspace(Bmin, Bmax, 11, endpoint=True)

#        self.BxNorm = colors.Normalize(vmin=Bxmin, vmax=Bxmax)
#        self.ByNorm = colors.Normalize(vmin=Bymin, vmax=Bymax)
        
        
        Vmin = 2. * min(self.diagnostics.Vx.min(), self.diagnostics.Vy.min(), -self.diagnostics.Vx.max(), -self.diagnostics.Vy.max())
        Vmax = 2. * max(self.diagnostics.Vx.max(), self.diagnostics.Vy.max(), -self.diagnostics.Vx.min(), -self.diagnostics.Vy.min())
        
#        Vxmin = self.diagnostics.Vx.min()
#        Vxmax = self.diagnostics.Vx.max()
#        
#        Vymin = self.diagnostics.Vy.min()
#        Vymax = self.diagnostics.Vy.max()

        divVmin = self.diagnostics.divV.min()
        divVmax = self.diagnostics.divV.max()
        
        if Vmin == Vmax:
            Vmin -= 1.
            Vmax += 1.
        
#        if Vxmin == Vxmax:
#            Vxmin -= 1.
#            Vxmax += 1.
#        
#        if Vymin == Vymax:
#            Vymin -= 1.
#            Vymax += 1.
        
        if divVmin == divVmax:
            divVmin -= 0.1
            divVmax += 0.1
        
        self.VxTicks = np.linspace(Vmin, Vmax, 11, endpoint=True)
        self.VyTicks = np.linspace(Vmin, Vmax, 11, endpoint=True)

        self.divVTicks = np.linspace(divVmin, divVmax, 11, endpoint=True)
        
        
        Pmin = min(self.diagnostics.e_magnetic.min(), -self.diagnostics.e_magnetic.max())
        Pmax = min(self.diagnostics.e_magnetic.max(), -self.diagnostics.e_magnetic.min())
        
        if Pmin == Pmax:
            Pmin -= 1.
            Pmax += 1.
        
        self.PTicks = np.linspace(Pmin, Pmax, 11, endpoint=True)


        Amin = min(self.diagnostics.A.min(), -self.diagnostics.A.max())
        Amax = max(self.diagnostics.A.max(), -self.diagnostics.A.min())
        Adif = Amax - Amin
        
#        self.ATicks = np.linspace(Amin + 0.3 * Adif, Amax + 0.2 * Adif, 21, endpoint=True)
        self.ATicks = np.linspace(Amin + 0.01 * Adif, Amax - 0.01 * Adif, 31)
        
    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0 or self.iTime-1 == self.nTime):
            return
        
#        self.update_boundaries()

        for ckey, cont in self.conts.items():
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
        
#         self.P[0:-1, 0:-1] = self.diagnostics.P[:,:]
#         self.P[  -1, 0:-1] = self.diagnostics.P[0,:]
#         self.P[   :,   -1] = self.P[:,0]
        
        self.P[0:-1, 0:-1] = self.diagnostics.e_magnetic[:,:]
        self.P[  -1, 0:-1] = self.diagnostics.e_magnetic[0,:]
        self.P[   :,   -1] = self.P[:,0]
        
        self.A[0:-1, 0:-1] = self.diagnostics.A[:,:]
        self.A[  -1, 0:-1] = self.diagnostics.A[0,:]
        self.A[   :,   -1] = self.A[:,0]
        
        
        Pmin = min(self.diagnostics.e_magnetic.min(), -self.diagnostics.e_magnetic.max())
        Pmax = max(self.diagnostics.e_magnetic.max(), -self.diagnostics.e_magnetic.min())
        
        if Pmin == Pmax:
            Pmin -= 1.
            Pmax += 1.
        
        self.PTicks = np.linspace(Pmin, Pmax, 11, endpoint=True)
        
                
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, self.BxTicks, extend='both')
#        self.cbars["Bx"] = self.figure.colorbar(self.conts["Bx"], ax=self.axes["Bx"], orientation='vertical', ticks=self.BxTicks)#, format='%0.2E'
       
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, self.ByTicks, extend='both')
#        self.cbars["By"] = self.figure.colorbar(self.conts["By"], ax=self.axes["By"], orientation='vertical', ticks=self.ByTicks)
        
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, self.VxTicks, extend='both')
#        self.cbars["Vx"] = self.figure.colorbar(self.conts["Vx"], ax=self.axes["Vx"], orientation='vertical', ticks=self.VxTicks)
        
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, self.VyTicks, extend='both')
#        self.cbars["Vy"] = self.figure.colorbar(self.conts["Vy"], ax=self.axes["Vy"], orientation='vertical', ticks=self.VyTicks)

#        self.conts["P"] = self.axes["P"].contourf(self.x, self.y, self.P.T, self.PTicks, cmap=cm.jet, extend='both')
        self.conts["P"] = self.axes["P"].contourf(self.x, self.y, self.P.T, self.PTicks, cmap=cm.jet, extend='both', ticks=self.PTicks)
        
#        self.conts["Babs"] = self.axes["Babs"].contourf(self.x, self.y, self.divV.T, self.divVTicks, cmap=cm.jet)
#        self.cbars["Babs"] = self.figure.colorbar(self.conts["Babs"], ax=self.axes["Babs"], orientation='vertical', ticks=self.divVTicks)
        
        
#        stream_n = 19
#        stream_res_fac = 10
#        stream_density = stream_n / 30. * stream_res_fac
#        self.stream_xstart = np.arange(stream_n, dtype=np.int) * stream_res_fac
#        self.stream_ystart = np.zeros(stream_n, dtype=np.int)
#        
#        self.stream_xstart[0:5 ] -= 5
#        self.stream_xstart[5:14] -= 1
#        self.stream_xstart[14: ] += 3
        
        
        ### temporarily disabled
#        self.axes["Babs"].clear()
#        plt.subplot(self.gs[0:10,1:3])
#        plt.streamplot(self.x, self.y, self.Bx.T, self.By.T, density=1.2, arrowsize=0., color='b')
#        plt.streamplot(self.x, self.y, self.Bx.T, self.By.T, density=1., arrowsize=0., color='b')
#        plt.streamplot(self.x, self.y, self.Bx.T, self.By.T, density=1., color='b')
#        plt.streamplot(self.x, self.y, self.Bx.T, self.By.T, density=.8, arrowsize=0., color='b')
#        plt.streamplot(self.x, self.y, self.Bx.T, self.By.T, density=.72, arrowsize=0., color='b')
#        plt.streamplot(self.x, self.y, self.Bx.T, self.By.T, density=.6, arrowsize=0., color='b')
#        plt.streamplot(self.x, self.y, self.Bx.T, self.By.T, xstart=self.stream_xstart[1:], ystart=self.stream_ystart[1:], density=stream_density, arrowsize=0., color='b')
        
        
        
#        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, self.BxTicks, cmap=cm.jet)
#        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, self.ByTicks, cmap=cm.jet)
#        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, self.VxTicks, cmap=cm.jet)
#        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, self.VyTicks, cmap=cm.jet)
        
#        self.cbars["By"].set_clim(self.ByTicks[0], self.ByTicks[-1])
#        self.cbars["By"].draw_all()
        
#        self.cbars["Bx"].set_ticks(self.BxTicks)
#        self.cbars["By"].set_ticks(self.ByTicks)
#        self.cbars["Vx"].set_ticks(self.VxTicks)
#        self.cbars["Vy"].set_ticks(self.VyTicks)
        
#        self.cbars["Bx"].set_clim(vmin=self.BxTicks[0], vmax=self.BxTicks[-1]) 
#        self.cbars["Bx"].draw_all()
#        
#        self.cbars["By"].set_clim(vmin=self.ByTicks[0], vmax=self.ByTicks[-1]) 
#        self.cbars["By"].draw_all()
        
#        self.conts["Babs"] = self.axes["Babs"].contourf(self.x, self.y, self.divV.T, self.divVTicks, cmap=cm.jet)
#        self.cbars["Babs"].set_ticks(self.divVTicks)
        
#        self.conts["Vabs"] = self.axes["Vabs"].contourf(self.x, self.y, self.divV.T, 10)        
        
#        if len(self.B[self.B == self.B[0,0]]) == len(self.B.ravel()):
#            self.conts["Babs"] = self.axes["Babs"].contourf(, 10)
#        else:
#            self.conts["Babs"] = self.axes["Babs"].contour(self.x, self.y, self.B.T, 10)
#        
#        self.conts["Vabs"] = self.axes["Vabs"].contourf(self.x, self.y, self.V.T, 20)
        
        
#        self.axes["Babs"].clear()
#        plt.subplot(self.gs[0:2,1:3])
#        streamplot(self.x, self.y, self.Bx.T, self.By.T, density=1.5, arrowsize=.5, color='b')
        
#        self.axes["Babs"].clear()
#        st_B = Streamlines(self.x, self.y, self.Bx.T, self.By.T, spacing=1)#, res=.25)
#        st_B.plot(ax=self.axes["Babs"])
        
#        self.axes["Vabs"].clear()
#        plt.subplot(self.gs[2:4,1:3])
#        streamplot(self.x, self.y, self.Vx.T, self.Vy.T, density=1, arrowsize=1, color='b')
        
        ### temporarily disabled
#        self.axes["Vabs"].clear()
#        self.axes["Vabs"].quiver(self.x[::2], self.y[::2], self.Bx.T[::2,::2], self.By.T[::2,::2])
        
#        self.conts["Vabs"] = self.axes["Vabs"].contour(self.x, self.y, self.A.T, self.PTicks, cmap=cm.jet, extend='both', ticks=self.PTicks)
#        self.conts["Vabs"] = self.axes["Vabs"].contour(self.x, self.y, self.A.T, 40)
#        self.conts["Vabs"] = self.axes["Vabs"].contour(self.x, self.y, self.A.T, self.ATicks, cmap=cm.jet)
        self.conts["Vabs"] = self.axes["Vabs"].contour(self.x, self.y, self.A.T, self.ATicks, extend='neither')
        
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
#        self.figure.canvas.draw()
#        self.figure.canvas.flush_events()
        plt.show(block=final)
        plt.pause(.001)
        
#        return self.figure
    
    
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
        
        
#        self.L1_magnetic[self.iTime] = self.diagnostics.L1_magnetic_error
#        self.L1_velocity[self.iTime] = self.diagnostics.L1_velocity_error
#        self.L2_magnetic[self.iTime] = self.diagnostics.L2_magnetic_error
#        self.L2_velocity[self.iTime] = self.diagnostics.L2_velocity_error
        
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
    
