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


class PlotEnergy(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, filename, ntMax=0, nPlot=1):
        '''
        Constructor
        '''
        
#        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', family='sans-serif', size='22')
        
        self.prefix = filename
        
        self.ntMax = diagnostics.nt
        
        if self.ntMax > ntMax and ntMax > 0:
            self.ntMax = ntMax
        
        self.nPlot = nPlot
        
        self.diagnostics = diagnostics
        
        
        self.energy      = np.zeros(self.ntMax+1)
        self.helicity    = np.zeros(self.ntMax+1)
        self.magnetic    = np.zeros(self.ntMax+1)
        
        
        print("")
        for i in range(0, self.ntMax+1):
            print("Reading timestep %5i" % (i))
            
            self.diagnostics.read_from_hdf5(i)
            self.diagnostics.update_invariants(i)
            
            if self.diagnostics.plot_energy:
                self.energy  [i] = self.diagnostics.energy
            else:
                self.energy  [i] = self.diagnostics.E_error
            
            if self.diagnostics.plot_helicity:
                self.helicity[i] = self.diagnostics.helicity
            else:
                self.helicity[i] = self.diagnostics.H_error
            
            if self.diagnostics.plot_magnetic:
                self.magnetic[i] = self.diagnostics.magnetic
            else:
                self.magnetic[i] = self.diagnostics.M_error
            
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        ## -> limit to 1.1f precision
        majorFormatter.set_powerlimits((-1,+1))
        majorFormatter.set_scientific(True)


        # set up figure for energy plot
        self.figure1 = plt.figure(num=1, figsize=(16,4))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.25, wspace=0.2)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.25)
        
        axesE = plt.subplot(1,1,1)
        axesE.plot(self.diagnostics.tGrid[0:ntMax+1:self.nPlot], self.energy[0:ntMax+1:self.nPlot])
        
        axesE.set_xlabel('$t$', labelpad=15, fontsize=26)
        axesE.set_xlim(self.diagnostics.tGrid[0], self.diagnostics.tGrid[ntMax])
        
        if self.diagnostics.plot_energy:
            axesE.set_ylabel('$E$', labelpad=15, fontsize=26)
        else:
            axesE.set_ylabel('$(E-E_0) / E_0$', labelpad=15, fontsize=26)
        
        axesE.yaxis.set_major_formatter(majorFormatter)
        
        for tick in axesE.xaxis.get_major_ticks():
            tick.set_pad(12)
        for tick in axesE.yaxis.get_major_ticks():
            tick.set_pad(8)
                
        plt.draw()
        
        filename = self.prefix + str('_energy_%06d' % self.ntMax) + '.png'
        plt.savefig(filename, dpi=300)
        filename = self.prefix + str('_energy_%06d' % self.ntMax) + '.pdf'
        plt.savefig(filename)
        
        
        # set up figure for helicity plot
        self.figure2 = plt.figure(num=2, figsize=(16,4))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.25, wspace=0.2)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.25)
        
        axesH = plt.subplot(1,1,1)
        axesH.plot(self.diagnostics.tGrid[0:ntMax+1:self.nPlot], self.helicity[0:ntMax+1:self.nPlot])
        axesH.set_xlim(self.diagnostics.tGrid[0], self.diagnostics.tGrid[ntMax])
        
        axesH.set_xlabel('$t$', labelpad=15, fontsize=26)
        
        if self.diagnostics.plot_helicity:
            axesH.set_ylabel('$H_{c}$', labelpad=15, fontsize=26)
        else:
            axesH.set_ylabel('$(H_{c}-H_{c,0}) / H_{c,0}$', labelpad=15, fontsize=26)
        
        axesH.yaxis.set_major_formatter(majorFormatter)
        
        for tick in axesH.xaxis.get_major_ticks():
            tick.set_pad(12)
        for tick in axesH.yaxis.get_major_ticks():
            tick.set_pad(8)
                
        plt.draw()
        
        filename = self.prefix + str('_c_helicity_%06d' % self.ntMax) + '.png'
        plt.savefig(filename, dpi=300)
        filename = self.prefix + str('_c_helicity_%06d' % self.ntMax) + '.pdf'
        plt.savefig(filename)
        
        

        # set up figure for helicity plot
        self.figure3 = plt.figure(num=3, figsize=(16,4))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.25, wspace=0.2)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.25)
        
        axesM = plt.subplot(1,1,1)
        axesM.plot(self.diagnostics.tGrid[0:ntMax+1:self.nPlot], self.magnetic[0:ntMax+1:self.nPlot])
        
        axesM.set_xlabel('$t$', labelpad=15, fontsize=26)
        axesM.set_xlim(self.diagnostics.tGrid[0], self.diagnostics.tGrid[ntMax])
        
        if self.diagnostics.plot_helicity:
            axesM.set_ylabel('$H_{m}$', labelpad=15, fontsize=26)
        else:
            axesM.set_ylabel('$(H_{m}-H_{m,0}) / H_{m,0}$', labelpad=15, fontsize=26)
        
        axesM.yaxis.set_major_formatter(majorFormatter)
        
        for tick in axesM.xaxis.get_major_ticks():
            tick.set_pad(12)
        for tick in axesM.yaxis.get_major_ticks():
            tick.set_pad(8)
                
        plt.draw()
        
        filename = self.prefix + str('_m_helicity_%06d' % self.ntMax) + '.png'
        plt.savefig(filename, dpi=300)
        filename = self.prefix + str('_m_helicity_%06d' % self.ntMax) + '.pdf'
        plt.savefig(filename)
        
        

    

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
    
    diagnostics = Diagnostics(args.hdf5_file)
    
    ntMax=args.ntmax
    nPlot=args.np
    
    if ntMax > 0 and ntMax < diagnostics.nt:
        nt = ntMax
    else:
        nt = diagnostics.nt
    
    plot  = PlotEnergy(diagnostics, args.hdf5_file.replace(".hdf5", ""), nt, nPlot)
    
    print
    print("Replay finished.")
    print
    
