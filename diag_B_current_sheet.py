'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse

import numpy as np

import matplotlib
#matplotlib.use('Cairo')
matplotlib.use('AGG')
#matplotlib.use('PDF')

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter


from imhd.diagnostics import Diagnostics 


class PlotMHD2D(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, filename, ntMax=0, nPlot=1, write=False):
        '''
        Constructor
        '''
        
#        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', family='sans-serif', size='28')
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        
        self.prefix = filename
        
        self.ntMax = diagnostics.nt
        
        if self.ntMax > ntMax and ntMax > 0:
            self.ntMax = ntMax
        
        self.nPlot = nPlot
        self.iTime = -1
        
        self.diagnostics = diagnostics
        
        
        self.x = np.zeros(diagnostics.nx+1)
        self.y = np.zeros(diagnostics.ny+1)
        
        self.x[0:-1] = self.diagnostics.xGrid
        self.x[  -1] = self.x[-2] + self.diagnostics.hx
        
        self.y[0:-1] = self.diagnostics.yGrid
        self.y[  -1] = self.y[-2] + self.diagnostics.hy
        
        self.A       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.J       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.PB      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(10,10))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.25, wspace=0.2)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.95, 't = 0.0', horizontalalignment='center', fontsize=30) 
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        ## -> limit to 1.1f precision
        majorFormatter.set_powerlimits((-1,+1))
        majorFormatter.set_scientific(True)

        # create axes
        self.axes = plt.subplot(1,1,1)
        
        # add data for zero timepoint and compute boundaries
        self.add_timepoint()
        self.update_boundaries()
        
        # create contour plot
        self.conts = self.axes.contour(self.x[1:-1], self.y[1:-1], self.A.T[1:-1,1:-1], self.ATicks, extend='min', colors='k')
#        self.conts = self.axes.contour(self.x[1:-1], self.y[1:-1], self.A.T[1:-1,1:-1], 20, norm=self.ANorm, extend='neither', color='k', linestyle='solid')
#        self.conts = self.axes.contour(self.x, self.y, self.PB.T, levels=self.PBTicks, extend='neither')
#        self.conts = self.axes.contourf(self.x, self.y, self.J.T, 51, norm=self.Jnorm)
        
        for tick in self.axes.xaxis.get_major_ticks():
            tick.set_pad(12)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.set_pad(8)
        
        
        # plot
        self.update()
        
    
    def read_data(self):
        
        self.A[0:-1, 0:-1] = self.diagnostics.A[:,:]
        self.A[  -1, 0:-1] = self.diagnostics.A[0,:]
        self.A[   :,   -1] = self.A[:,0]
        
    
    def update_boundaries(self):
        
        Amin = min(np.array(self.diagnostics.A).min(), -np.array(self.diagnostics.A).max())
        Amax = max(np.array(self.diagnostics.A).max(), -np.array(self.diagnostics.A).min())
        Adiff = Amax - Amin
        
#        self.Anorm = colors.Normalize(vmin=Amin - 0.2*Adiff, vmax=Amax + 0.2*Adiff)
#        self.ANorm  = colors.Normalize(vmin=Amin - 0.1 * Adiff, vmax=Amax - 0.05 * Adiff)
#        self.ATicks = np.linspace(Amin + 0.01 * Adiff, Amax - 0.01 * Adiff, 31)
#        self.ATicks = np.linspace(Amin + 0.02 * Adiff, Amax - 0.02 * Adiff, 51, endpoint=True)
        self.ATicks = np.linspace(-1.3543, -0.07, 17, endpoint=True)
#        self.ATicks = np.linspace(-1.353, -0.07, 17, endpoint=True)
#        self.ATicks = np.linspace(-0.872, -0.551, 17, endpoint=True)

    
    
    def update(self):
        
        if not (self.iTime == 0 or (self.iTime) % self.nPlot == 0 or self.iTime == self.ntMax):
            return
        
        self.read_data()

        for coll in self.conts.collections:
            self.axes.collections.remove(coll)
        
        self.conts = self.axes.contour(self.x[1:-1], self.y[1:-1], self.A.T[1:-1,1:-1], self.ATicks, extend='min', colors='k')
#        self.conts = self.axes.contour(self.x[1:-1], self.y[1:-1], self.A.T[1:-1,1:-1], 20, norm=self.ANorm, extend='neither', color='k', linestyle='solid')
#        self.conts = self.axes.contour(self.x, self.y, self.PB.T, levels=self.PBTicks, extend='neither')
#        self.conts = self.axes.contourf(self.x, self.y, self.J.T, 51, norm=self.Jnorm)
#        plt.clabel(self.conts, inline=1, fontsize=10)
        
        plt.draw()
        
#        filename = self.prefix + str('_B_%06d' % self.iTime) + '.png'
#        plt.savefig(filename, dpi=300)
        filename = self.prefix + str('_B_%06d' % self.iTime) + '.pdf'
        plt.savefig(filename)
    
    
    def add_timepoint(self):
        self.iTime += 1
        self.title.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        
    


class Plot(object):
    '''
    
    '''


    def __init__(self, hdf5_file, nPlot=1, ntMax=0):
        '''
        Constructor
        '''
        
        self.diagnostics = Diagnostics(hdf5_file)
        
        if ntMax > 0 and ntMax < self.diagnostics.nt:
            self.nt = ntMax
        else:
            self.nt = self.diagnostics.nt
        
        self.plot = PlotMHD2D(self.diagnostics, args.hdf5_file.replace(".hdf5", ""), self.nt, nPlot)
        
    
    def update(self, itime):
        self.diagnostics.read_from_hdf5(itime)
        self.diagnostics.update_invariants(itime)
        
        if itime > 0:
            self.plot.add_timepoint()
        
        self.plot.update()
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            print("it = %4i" % (itime))
            self.update(itime)
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ideal MHD Solver in 2D :: Current Sheet Diagnostics')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')
    parser.add_argument('-ntmax', metavar='i', type=int, default=0,
                        help='limit to i points in time')
    
    args = parser.parse_args()
    
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = Plot(args.hdf5_file, ntMax=args.ntmax, nPlot=args.np)
    pyvp.run()
    
    print
    print("Replay finished.")
    print
    
