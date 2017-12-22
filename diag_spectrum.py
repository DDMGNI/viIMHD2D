'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse

import numpy as np
from numpy.fft import *

import matplotlib
#matplotlib.use('Cairo')
matplotlib.use('AGG')
#matplotlib.use('PDF')

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter, ScalarFormatter
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
        matplotlib.rc('font', family='sans-serif', size='24')
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        matplotlib.rcParams['grid.linestyle'] = "dotted"
        
        self.prefix = filename
        
        self.ntMax = diagnostics.nt
        
        if self.ntMax > ntMax and ntMax > 0:
            self.ntMax = ntMax
        
        self.nPlot = nPlot
        self.iTime = -1
        
        self.diagnostics = diagnostics
        
        self.kx = fftshift(fftfreq(diagnostics.nx+1, diagnostics.hx))
        self.ky = fftshift(fftfreq(diagnostics.ny+1, diagnostics.hy))
        
        self.Bx = np.zeros((diagnostics.nx, diagnostics.ny))
        self.By = np.zeros((diagnostics.nx, diagnostics.ny))
        self.Vx = np.zeros((diagnostics.nx, diagnostics.ny))
        self.Vy = np.zeros((diagnostics.nx, diagnostics.ny))
        
        self.BxSpectrum = np.zeros((diagnostics.nx, diagnostics.ny))
        self.BySpectrum = np.zeros((diagnostics.nx, diagnostics.ny))
        self.VxSpectrum = np.zeros((diagnostics.nx, diagnostics.ny))
        self.VySpectrum = np.zeros((diagnostics.nx, diagnostics.ny))
        
        
        # set up figure/window for Bx
        self.figure_Bx = plt.figure(num=1, figsize=(12,10))
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.12)
        
        # set up plot title
        self.title_Bx = self.figure_Bx.text(0.5, 0.93, 't = 0.0', horizontalalignment='center', fontsize=30)

        # create axes
        self.axes_Bx = plt.subplot(1,1,1)
        self.axes_Bx.set_xlabel('$k_x$', labelpad=15, fontsize=24)
        self.axes_Bx.set_ylabel('$k_y$', labelpad=15, fontsize=24)
        self.axes_Bx.set_xlim(self.kx[0], self.kx[-1])
        self.axes_Bx.set_ylim(self.ky[0], self.ky[-1])

        # create plot
        self.plot_Bx = self.axes_Bx.pcolormesh(self.kx, self.ky, self.BxSpectrum.T)
        
        # add colorbar
        divider = make_axes_locatable(self.axes_Bx)
        cax_Bx = divider.append_axes('right', size='5%', pad=0.1)
        self.figure_Bx.colorbar(self.plot_Bx, cax=cax_Bx, orientation='vertical')        

        
        # set up figure/window for By
        self.figure_By = plt.figure(num=2, figsize=(12,10))
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.12)
        
        # set up plot title
        self.title_By = self.figure_By.text(0.5, 0.93, 't = 0.0', horizontalalignment='center', fontsize=30)

        # create axes
        self.axes_By = plt.subplot(1,1,1)
        self.axes_By.set_xlabel('$k_x$', labelpad=15, fontsize=24)
        self.axes_By.set_ylabel('$k_y$', labelpad=15, fontsize=24)
        self.axes_By.set_xlim(self.kx[0], self.kx[-1])
        self.axes_By.set_ylim(self.ky[0], self.ky[-1])

        # create plot
        self.plot_By = self.axes_By.pcolormesh(self.kx, self.ky, self.BySpectrum.T)
        
        # add colorbar
        divider = make_axes_locatable(self.axes_By)
        cax_By = divider.append_axes('right', size='5%', pad=0.1)
        self.figure_By.colorbar(self.plot_By, cax=cax_By, orientation='vertical')        

        
        # set up figure/window for Vx
        self.figure_Vx = plt.figure(num=3, figsize=(12,10))
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.12)
        
        # set up plot title
        self.title_Vx = self.figure_Vx.text(0.5, 0.93, 't = 0.0', horizontalalignment='center', fontsize=30)

        # create axes
        self.axes_Vx = plt.subplot(1,1,1)
        self.axes_Vx.set_xlabel('$k_x$', labelpad=15, fontsize=24)
        self.axes_Vx.set_ylabel('$k_y$', labelpad=15, fontsize=24)
        self.axes_Vx.set_xlim(self.kx[0], self.kx[-1])
        self.axes_Vx.set_ylim(self.ky[0], self.ky[-1])

        # create plot
        self.plot_Vx = self.axes_Vx.pcolormesh(self.kx, self.ky, self.VxSpectrum.T)
        
        # add colorbar
        divider = make_axes_locatable(self.axes_Vx)
        cax_Vx = divider.append_axes('right', size='5%', pad=0.1)
        self.figure_Vx.colorbar(self.plot_Vx, cax=cax_Vx, orientation='vertical')        

        
        # set up figure/window for Vy
        self.figure_Vy = plt.figure(num=4, figsize=(12,10))
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.12)
        
        # set up plot title
        self.title_Vy = self.figure_Vy.text(0.5, 0.93, 't = 0.0', horizontalalignment='center', fontsize=30)

        # create axes
        self.axes_Vy = plt.subplot(1,1,1)
        self.axes_Vy.set_xlabel('$k_x$', labelpad=15, fontsize=24)
        self.axes_Vy.set_ylabel('$k_y$', labelpad=15, fontsize=24)
        self.axes_Vy.set_xlim(self.kx[0], self.kx[-1])
        self.axes_Vy.set_ylim(self.ky[0], self.ky[-1])

        # create plot
        self.plot_Vy = self.axes_Vy.pcolormesh(self.kx, self.ky, self.VySpectrum.T)
        
        # add colorbar
        divider = make_axes_locatable(self.axes_Vy)
        cax_Vy = divider.append_axes('right', size='5%', pad=0.1)
        self.figure_Vy.colorbar(self.plot_Vy, cax=cax_Vy, orientation='vertical')        

        
        # add data for zero timepoint and compute boundaries
        self.add_timepoint()
        
        
        # plot
        self.update()
        
    
    def read_data(self):
        
        self.Bx[:,:] = self.diagnostics.Bx
        self.By[:,:] = self.diagnostics.By
        self.Vx[:,:] = self.diagnostics.Vx
        self.Vy[:,:] = self.diagnostics.Vy
        
        self.BxSpectrum[:,:] = np.abs(fftshift(fft2(self.Bx)))
        self.BySpectrum[:,:] = np.abs(fftshift(fft2(self.By)))
        self.VxSpectrum[:,:] = np.abs(fftshift(fft2(self.Vx)))
        self.VySpectrum[:,:] = np.abs(fftshift(fft2(self.Vy)))
        
    
    
    def update(self):
        
        if not (self.iTime == 0 or (self.iTime) % self.nPlot == 0 or self.iTime == self.ntMax):
            return
        
        self.read_data()

        self.plot_Bx.set_array(self.BxSpectrum.T.ravel())
        self.plot_By.set_array(self.BySpectrum.T.ravel())
        self.plot_Vx.set_array(self.VxSpectrum.T.ravel())
        self.plot_Vy.set_array(self.VySpectrum.T.ravel())
        
        self.figure_Bx.savefig(self.prefix + str('_spectrum_Bx_%06d' % self.iTime) + '.png', dpi=100)
        self.figure_By.savefig(self.prefix + str('_spectrum_By_%06d' % self.iTime) + '.png', dpi=100)
        self.figure_Vx.savefig(self.prefix + str('_spectrum_Vx_%06d' % self.iTime) + '.png', dpi=100)
        self.figure_Vy.savefig(self.prefix + str('_spectrum_Vy_%06d' % self.iTime) + '.png', dpi=100)
        

    
    
    def add_timepoint(self):
        self.iTime += 1
        self.title_Bx.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_By.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_Vx.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
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
    parser = argparse.ArgumentParser(description='Vlasov-Poisson Solver in 1D')
    
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
    
