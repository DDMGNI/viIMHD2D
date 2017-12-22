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
        
        self.BxPhase    = np.zeros((diagnostics.nx, diagnostics.ny))
        self.ByPhase    = np.zeros((diagnostics.nx, diagnostics.ny))
        self.VxPhase    = np.zeros((diagnostics.nx, diagnostics.ny))
        self.VyPhase    = np.zeros((diagnostics.nx, diagnostics.ny))


        # compute initial phase
        
        self.Bx[:,:] = self.diagnostics.Bx
        self.By[:,:] = self.diagnostics.By
        self.Vx[:,:] = self.diagnostics.Vx
        self.Vy[:,:] = self.diagnostics.Vy
        
        BxFft = fftshift(fft2(self.Bx))
        ByFft = fftshift(fft2(self.By))
        VxFft = fftshift(fft2(self.Vx))
        VyFft = fftshift(fft2(self.Vy))
        
        self.BxPhase0 = np.angle(BxFft)
        self.ByPhase0 = np.angle(ByFft)
        self.VxPhase0 = np.angle(VxFft)
        self.VyPhase0 = np.angle(VyFft)

        
        self.read_data()
        
        
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
        self.plot_Bx = self.axes_Bx.pcolormesh(self.kx, self.ky, self.BxSpectrum.T, cmap=plt.get_cmap('viridis'), vmin=0, vmax=self.BxSpectrum.max())
        
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
        self.plot_By = self.axes_By.pcolormesh(self.kx, self.ky, self.BySpectrum.T, cmap=plt.get_cmap('viridis'), vmin=0, vmax=self.BySpectrum.max())
        
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
        self.plot_Vx = self.axes_Vx.pcolormesh(self.kx, self.ky, self.VxSpectrum.T, cmap=plt.get_cmap('viridis'), vmin=0, vmax=self.VxSpectrum.max())
        
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
        self.plot_Vy = self.axes_Vy.pcolormesh(self.kx, self.ky, self.VySpectrum.T, cmap=plt.get_cmap('viridis'), vmin=0, vmax=self.VySpectrum.max())
        
        # add colorbar
        divider = make_axes_locatable(self.axes_Vy)
        cax_Vy = divider.append_axes('right', size='5%', pad=0.1)
        self.figure_Vy.colorbar(self.plot_Vy, cax=cax_Vy, orientation='vertical')        

        
        
        # set up figure/window for Bx
        self.figure_phase_Bx = plt.figure(num=1, figsize=(12,10))
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.12)
        
        # set up plot title
        self.title_phase_Bx = self.figure_phase_Bx.text(0.5, 0.93, 't = 0.0', horizontalalignment='center', fontsize=30)

        # create axes
        self.axes_phase_Bx = plt.subplot(1,1,1)
        self.axes_phase_Bx.set_xlabel('$k_x$', labelpad=15, fontsize=24)
        self.axes_phase_Bx.set_ylabel('$k_y$', labelpad=15, fontsize=24)
        self.axes_phase_Bx.set_xlim(self.kx[0], self.kx[-1])
        self.axes_phase_Bx.set_ylim(self.ky[0], self.ky[-1])

        # create plot
        self.plot_phase_Bx = self.axes_phase_Bx.pcolormesh(self.kx, self.ky, self.BxPhase.T, cmap=plt.get_cmap('viridis'), vmin=self.BxPhase.min(), vmax=self.BxPhase.max())
        
        # add colorbar
        divider = make_axes_locatable(self.axes_phase_Bx)
        cax_phase_Bx = divider.append_axes('right', size='5%', pad=0.1)
        self.figure_phase_Bx.colorbar(self.plot_phase_Bx, cax=cax_phase_Bx, orientation='vertical')        

        
        # set up figure/window for By
        self.figure_phase_By = plt.figure(num=2, figsize=(12,10))
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.12)
        
        # set up plot title
        self.title_phase_By = self.figure_phase_By.text(0.5, 0.93, 't = 0.0', horizontalalignment='center', fontsize=30)

        # create axes
        self.axes_phase_By = plt.subplot(1,1,1)
        self.axes_phase_By.set_xlabel('$k_x$', labelpad=15, fontsize=24)
        self.axes_phase_By.set_ylabel('$k_y$', labelpad=15, fontsize=24)
        self.axes_phase_By.set_xlim(self.kx[0], self.kx[-1])
        self.axes_phase_By.set_ylim(self.ky[0], self.ky[-1])

        # create plot
        self.plot_phase_By = self.axes_phase_By.pcolormesh(self.kx, self.ky, self.ByPhase.T, cmap=plt.get_cmap('viridis'), vmin=self.ByPhase.min(), vmax=self.ByPhase.max())
        
        # add colorbar
        divider = make_axes_locatable(self.axes_phase_By)
        cax_phase_By = divider.append_axes('right', size='5%', pad=0.1)
        self.figure_phase_By.colorbar(self.plot_phase_By, cax=cax_phase_By, orientation='vertical')        

        
        # set up figure/window for Vx
        self.figure_phase_Vx = plt.figure(num=3, figsize=(12,10))
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.12)
        
        # set up plot title
        self.title_phase_Vx = self.figure_phase_Vx.text(0.5, 0.93, 't = 0.0', horizontalalignment='center', fontsize=30)

        # create axes
        self.axes_phase_Vx = plt.subplot(1,1,1)
        self.axes_phase_Vx.set_xlabel('$k_x$', labelpad=15, fontsize=24)
        self.axes_phase_Vx.set_ylabel('$k_y$', labelpad=15, fontsize=24)
        self.axes_phase_Vx.set_xlim(self.kx[0], self.kx[-1])
        self.axes_phase_Vx.set_ylim(self.ky[0], self.ky[-1])

        # create plot
        self.plot_phase_Vx = self.axes_phase_Vx.pcolormesh(self.kx, self.ky, self.VxPhase.T, cmap=plt.get_cmap('viridis'), vmin=self.VxPhase.min(), vmax=self.VxPhase.max())
        
        # add colorbar
        divider = make_axes_locatable(self.axes_phase_Vx)
        cax_phase_Vx = divider.append_axes('right', size='5%', pad=0.1)
        self.figure_phase_Vx.colorbar(self.plot_phase_Vx, cax=cax_phase_Vx, orientation='vertical')        

        
        # set up figure/window for Vy
        self.figure_phase_Vy = plt.figure(num=4, figsize=(12,10))
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.12)
        
        # set up plot title
        self.title_phase_Vy = self.figure_phase_Vy.text(0.5, 0.93, 't = 0.0', horizontalalignment='center', fontsize=30)

        # create axes
        self.axes_phase_Vy = plt.subplot(1,1,1)
        self.axes_phase_Vy.set_xlabel('$k_x$', labelpad=15, fontsize=24)
        self.axes_phase_Vy.set_ylabel('$k_y$', labelpad=15, fontsize=24)
        self.axes_phase_Vy.set_xlim(self.kx[0], self.kx[-1])
        self.axes_phase_Vy.set_ylim(self.ky[0], self.ky[-1])

        # create plot
        self.plot_phase_Vy = self.axes_phase_Vy.pcolormesh(self.kx, self.ky, self.VyPhase.T, cmap=plt.get_cmap('viridis'), vmin=self.VyPhase.min(), vmax=self.VyPhase.max())
        
        # add colorbar
        divider = make_axes_locatable(self.axes_phase_Vy)
        cax_phase_Vy = divider.append_axes('right', size='5%', pad=0.1)
        self.figure_phase_Vy.colorbar(self.plot_phase_Vy, cax=cax_phase_Vy, orientation='vertical')        

        
        # add data for zero timepoint and compute boundaries
        self.add_timepoint()
        
        
        # plot
        self.update()
        
    
    def read_data(self):
        
        self.Bx[:,:] = self.diagnostics.Bx
        self.By[:,:] = self.diagnostics.By
        self.Vx[:,:] = self.diagnostics.Vx
        self.Vy[:,:] = self.diagnostics.Vy
        
        BxFft = fftshift(fft2(self.Bx))
        ByFft = fftshift(fft2(self.By))
        VxFft = fftshift(fft2(self.Vx))
        VyFft = fftshift(fft2(self.Vy))
        
        self.BxSpectrum[:,:] = np.abs(BxFft)
        self.BySpectrum[:,:] = np.abs(ByFft)
        self.VxSpectrum[:,:] = np.abs(VxFft)
        self.VySpectrum[:,:] = np.abs(VyFft)
        
        self.BxPhase[:,:] = np.angle(BxFft) - self.BxPhase0
        self.ByPhase[:,:] = np.angle(ByFft) - self.ByPhase0
        self.VxPhase[:,:] = np.angle(VxFft) - self.VxPhase0
        self.VyPhase[:,:] = np.angle(VyFft) - self.VyPhase0
        
    
    
    def update(self):
        
        if not (self.iTime == 0 or (self.iTime) % self.nPlot == 0 or self.iTime == self.ntMax):
            return
        
        self.read_data()

        self.plot_Bx.set_array(self.BxSpectrum.T.ravel())
        self.plot_By.set_array(self.BySpectrum.T.ravel())
        self.plot_Vx.set_array(self.VxSpectrum.T.ravel())
        self.plot_Vy.set_array(self.VySpectrum.T.ravel())
        
        self.plot_phase_Bx.set_array(self.BxPhase.T.ravel())
        self.plot_phase_By.set_array(self.ByPhase.T.ravel())
        self.plot_phase_Vx.set_array(self.VxPhase.T.ravel())
        self.plot_phase_Vy.set_array(self.VyPhase.T.ravel())
        
        self.figure_Bx.savefig(self.prefix + str('_spectrum_Bx_%06d' % self.iTime) + '.png', dpi=100)
        self.figure_By.savefig(self.prefix + str('_spectrum_By_%06d' % self.iTime) + '.png', dpi=100)
        self.figure_Vx.savefig(self.prefix + str('_spectrum_Vx_%06d' % self.iTime) + '.png', dpi=100)
        self.figure_Vy.savefig(self.prefix + str('_spectrum_Vy_%06d' % self.iTime) + '.png', dpi=100)
        
        self.figure_phase_Bx.savefig(self.prefix + str('_phase_Bx_%06d' % self.iTime) + '.png', dpi=100)
        self.figure_phase_By.savefig(self.prefix + str('_phase_By_%06d' % self.iTime) + '.png', dpi=100)
        self.figure_phase_Vx.savefig(self.prefix + str('_phase_Vx_%06d' % self.iTime) + '.png', dpi=100)
        self.figure_phase_Vy.savefig(self.prefix + str('_phase_Vy_%06d' % self.iTime) + '.png', dpi=100)
        

    
    
    def add_timepoint(self):
        self.iTime += 1
        self.title_Bx.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_By.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_Vx.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_Vy.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_phase_Bx.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_phase_By.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_phase_Vx.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        self.title_phase_Vy.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
    


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
    
