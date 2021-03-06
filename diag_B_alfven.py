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
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        matplotlib.rc('font', family='sans-serif', size='22')
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        matplotlib.rcParams['grid.linestyle'] = "dotted"
        
        self.prefix = filename
        
        self.ntMax = diagnostics.nt
        
        if self.ntMax > ntMax and ntMax > 0:
            self.ntMax = ntMax
        
        self.nPlot = nPlot
        self.iTime = -1
        
        self.diagnostics = diagnostics
        
        
        self.t  = np.array(self.diagnostics.tGrid)

        self.x       = np.zeros(diagnostics.nx+2)
        self.x[1:-1] = np.array(self.diagnostics.xGrid) + 0.5 * self.diagnostics.hx
        self.x[   0] = self.x[ 1] - self.diagnostics.hx
        self.x[  -1] = self.x[-2] + self.diagnostics.hx
        
        self.By = np.zeros(diagnostics.nx+2)
        self.Vy = np.zeros(diagnostics.nx+2)
        
        self.xTrace  = np.zeros(diagnostics.nx+1)
        self.ByTrace = np.zeros((diagnostics.nx+1, diagnostics.nt+1))
        self.VyTrace = np.zeros((diagnostics.nx+1, diagnostics.nt+1))
        
        self.xTrace[0:-1] = self.diagnostics.xGrid
        self.xTrace[  -1] = self.diagnostics.xGrid[-1] + self.diagnostics.hx
        
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        majorFormatter.set_powerlimits((-1,+1))# -> limit to 1.1f precision
        majorFormatter.set_scientific(True)


        # set up figure/window for By
        self.figure_By = plt.figure(num=1, figsize=(10,5))
        plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.2)
        
        # set up plot title
        self.title_By = self.figure_By.text(0.5, 0.9, 't = 0.0', horizontalalignment='center', fontsize=28)

        # create axes
        self.axes_By = plt.subplot(1,1,1)
        self.axes_By.set_xlabel('$x$', labelpad=15, fontsize=24)
        self.axes_By.set_ylabel('$B_y(x)$', labelpad=15, fontsize=24)
        self.axes_By.set_xlim(self.diagnostics.xMin, self.diagnostics.xMax)
        self.axes_By.set_ylim(np.array(self.diagnostics.By[:,self.diagnostics.ny//2]).min(), np.array(self.diagnostics.By[:,self.diagnostics.ny//2]).max())
        self.axes_By.yaxis.set_major_formatter(majorFormatter)

        # add grid
        plt.grid()
        
        # create plot
        self.plot_By, = self.axes_By.plot(self.x, self.By)
        

        # set up figure/window for Vy
        self.figure_Vy = plt.figure(num=2, figsize=(10,5))
        plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.2)
        
        # set up plot title
        self.title_Vy = self.figure_Vy.text(0.5, 0.9, 't = 0.0', horizontalalignment='center', fontsize=28)

        # create axes
        self.axes_Vy = plt.subplot(1,1,1)
        self.axes_Vy.set_xlabel('$x$', labelpad=15, fontsize=24)
        self.axes_Vy.set_ylabel('$V_y(x)$', labelpad=15, fontsize=24)
        self.axes_Vy.set_xlim(self.diagnostics.xMin, self.diagnostics.xMax)
        self.axes_Vy.set_ylim(np.array(self.diagnostics.Vy[:,self.diagnostics.ny//2]).min(), np.array(self.diagnostics.Vy[:,self.diagnostics.ny//2]).max())
        self.axes_Vy.yaxis.set_major_formatter(majorFormatter)
        
        # add grid
        plt.grid()
        
        # create plot
        self.plot_Vy, = self.axes_Vy.plot(self.x, self.Vy)
        
        
        # add data for zero timepoint and compute boundaries
        self.add_timepoint()
        
        
        # plot
        self.update()
        
    
    def read_data(self):
        
        self.By[1:-1] = self.diagnostics.By[ :,self.diagnostics.ny//2]
        self.By[   0] = self.diagnostics.By[-1,self.diagnostics.ny//2]
        self.By[  -1] = self.diagnostics.By[ 0,self.diagnostics.ny//2]
        
        self.Vy[1:-1] = self.diagnostics.Vy[ :,self.diagnostics.ny//2]
        self.Vy[   0] = self.diagnostics.Vy[-1,self.diagnostics.ny//2]
        self.Vy[  -1] = self.diagnostics.Vy[ 0,self.diagnostics.ny//2]
        
        self.ByTrace[:, self.iTime] = self.By[1:]
        self.VyTrace[:, self.iTime] = self.Vy[1:]
    
    
    def update(self):
        
        if not (self.iTime == 0 or (self.iTime) % self.nPlot == 0 or self.iTime == self.ntMax):
            return
        
        self.read_data()

        self.plot_By.set_ydata(self.By)
        self.plot_Vy.set_ydata(self.Vy)
        
        self.figure_By.savefig(self.prefix + str('_By_%06d' % self.iTime) + '.pdf')
        self.figure_Vy.savefig(self.prefix + str('_Vy_%06d' % self.iTime) + '.pdf')
        

        if self.iTime == self.ntMax:
            # create ByTrace figure
            figure_ByTrace, axes_ByTrace = plt.subplots(num=3, figsize=(16,10))
            plt.subplots_adjust(left=0.1, right=0.88, top=0.95, bottom=0.1)
            
            axes_ByTrace.set_xlabel('$t$', labelpad=15, fontsize=24)
            axes_ByTrace.set_ylabel('$x$', labelpad=15, fontsize=24)

            pcms_By = axes_ByTrace.pcolormesh(self.t, self.xTrace, self.ByTrace, cmap=plt.get_cmap('viridis'))
            axes_ByTrace.set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[-1]))
            axes_ByTrace.set_ylim((self.diagnostics.xGrid[0], self.diagnostics.xGrid[-1]))

            divider_By = make_axes_locatable(axes_ByTrace)
            cax_By = divider_By.append_axes('right', size='5%', pad=0.1)
            figure_ByTrace.colorbar(pcms_By, cax=cax_By, orientation='vertical')

            figure_ByTrace.savefig(self.prefix + str('_ByTrace.png'), dpi=100)
        
    
    
            # create VyTrace figure
            figure_VyTrace, axes_VyTrace = plt.subplots(num=4, figsize=(16,10))
            plt.subplots_adjust(left=0.1, right=0.88, top=0.95, bottom=0.1)
            
            axes_VyTrace.set_xlabel('$t$', labelpad=15, fontsize=24)
            axes_VyTrace.set_ylabel('$x$', labelpad=15, fontsize=24)
            
            pcms_Vy = axes_VyTrace.pcolormesh(self.t, self.xTrace, self.VyTrace, cmap=plt.get_cmap('viridis'))
            axes_VyTrace.set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[-1]))
            axes_VyTrace.set_ylim((self.diagnostics.xGrid[0], self.diagnostics.xGrid[-1]))

            divider_Vy = make_axes_locatable(axes_VyTrace)
            cax_Vy = divider_Vy.append_axes('right', size='5%', pad=0.1)
            figure_VyTrace.colorbar(pcms_Vy, cax=cax_Vy, orientation='vertical')

            figure_VyTrace.savefig(self.prefix + str('_VyTrace.png'), dpi=100)
    
    
    def add_timepoint(self):
        self.iTime += 1
        self.title_By.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_Vy.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        
    


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
    parser = argparse.ArgumentParser(description='Ideal MHD Solver in 2D :: Alfven Wave Diagnostics')
    
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
    
