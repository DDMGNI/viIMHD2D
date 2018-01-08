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
from matplotlib.colors import LinearSegmentedColormap
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
        
        
        _Paired_data = {'blue': [(0.0, 0.89019608497619629,
            0.89019608497619629), (0.090909090909090912, 0.70588237047195435,
    0.70588237047195435), (0.18181818181818182, 0.54117649793624878,
    0.54117649793624878), (0.27272727272727271, 0.17254902422428131,
    0.17254902422428131), (0.36363636363636365, 0.60000002384185791,
    0.60000002384185791), (0.45454545454545453, 0.10980392247438431,
    0.10980392247438431), (0.54545454545454541, 0.43529412150382996,
    0.43529412150382996), (0.63636363636363635, 0.0, 0.0),
    (0.72727272727272729, 0.83921569585800171, 0.83921569585800171),
    (0.81818181818181823, 0.60392159223556519, 0.60392159223556519),
    (0.90909090909090906, 0.60000002384185791, 0.60000002384185791), (1.0,
    0.15686275064945221, 0.15686275064945221)],

    'green': [(0.0, 0.80784314870834351, 0.80784314870834351),
    (0.090909090909090912, 0.47058823704719543, 0.47058823704719543),
    (0.18181818181818182, 0.87450981140136719, 0.87450981140136719),
    (0.27272727272727271, 0.62745100259780884, 0.62745100259780884),
    (0.36363636363636365, 0.60392159223556519, 0.60392159223556519),
    (0.45454545454545453, 0.10196078568696976, 0.10196078568696976),
    (0.54545454545454541, 0.74901962280273438, 0.74901962280273438),
    (0.63636363636363635, 0.49803921580314636, 0.49803921580314636),
    (0.72727272727272729, 0.69803923368453979, 0.69803923368453979),
    (0.81818181818181823, 0.23921568691730499, 0.23921568691730499),
    (0.90909090909090906, 1.0, 1.0), (1.0, 0.3490196168422699,
    0.3490196168422699)],

    'red': [(0.0, 0.65098041296005249, 0.65098041296005249),
    (0.090909090909090912, 0.12156862765550613, 0.12156862765550613),
    (0.18181818181818182, 0.69803923368453979, 0.69803923368453979),
    (0.27272727272727271, 0.20000000298023224, 0.20000000298023224),
    (0.36363636363636365, 0.9843137264251709, 0.9843137264251709),
    (0.45454545454545453, 0.89019608497619629, 0.89019608497619629),
    (0.54545454545454541, 0.99215686321258545, 0.99215686321258545),
    (0.63636363636363635, 1.0, 1.0), (0.72727272727272729,
    0.7921568751335144, 0.7921568751335144), (0.81818181818181823,
    0.41568627953529358, 0.41568627953529358), (0.90909090909090906,
    1.0, 1.0), (1.0, 0.69411766529083252, 0.69411766529083252)]}
        
        
        paired_linear = LinearSegmentedColormap('PairedLin', _Paired_data, 256)
        plt.register_cmap(cmap=paired_linear)
        
        
#        matplotlib.rc('text', usetex=True)
#         matplotlib.rc('font', family='sans-serif', size='28')
        
        self.prefix = filename
        
        self.ntMax = diagnostics.nt
        
        if self.ntMax > ntMax and ntMax > 0:
            self.ntMax = ntMax
        
        self.nPlot = nPlot
        self.iTime = -1
        
        self.diagnostics = diagnostics
        
        
        self.x = np.zeros(diagnostics.nx+1)
        self.y = np.zeros(diagnostics.ny+1)
        
        self.xpc = np.zeros(diagnostics.nx+2)
        self.ypc = np.zeros(diagnostics.ny+2)
        
        self.x[0:-1] = self.diagnostics.xGrid
        self.x[  -1] = self.x[-2] + self.diagnostics.hx
        
        self.y[0:-1] = self.diagnostics.yGrid
        self.y[  -1] = self.y[-2] + self.diagnostics.hy
        
        self.xpc[0:-1] = self.x
        self.xpc[  -1] = self.xpc[-2] + self.diagnostics.hx
        self.xpc[:] -= 0.5 * self.diagnostics.hx
        
        self.ypc[0:-1] = self.y
        self.ypc[  -1] = self.ypc[-2] + self.diagnostics.hy
        self.ypc[:] -= 0.5 * self.diagnostics.hy
        
        self.A       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Ai      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.J       = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.PB      = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        
        
        self.read_data()
        self.update_boundaries()
        
        # set up figure/window size
        self.figure, self.axes = plt.subplots(num=None, figsize=(10,10))
        self.figure.tight_layout()
        
        # set up plot margins
#         plt.subplots_adjust(hspace=0.25, wspace=0.2)
#         plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        # set up plot title
#         self.title = self.figure.text(0.5, 0.95, 't = 0.0' % (diagnostics.tGrid[self.iTime]), horizontalalignment='center', fontsize=30) 
        
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
        
        # create current density plot
#        self.conts = self.axes.contourf(self.x, self.y, self.Ai.T, 51, cmap=plt.get_cmap('Paired'))
        self.pcm_J = self.axes.pcolormesh(self.xpc, self.ypc, self.Ai.T, cmap='PairedLin')
        self.axes.set_xlim((self.x[0],self.x[-1])) 
        self.axes.set_ylim((self.y[0],self.y[-1])) 
        
#         for tick in self.axes.xaxis.get_major_ticks():
#             tick.set_pad(12)
#         for tick in self.axes.yaxis.get_major_ticks():
#             tick.set_pad(8)
        
        
        # plot
        self.update()
        
    
    def read_data(self):
        
        self.A[0:-1, 0:-1] = self.diagnostics.A[:,:]
        self.A[  -1, 0:-1] = self.diagnostics.A[0,:]
        self.A[   :,   -1] = self.A[:,0]
        
        self.Ai[0:-1, 0:-1] = -self.diagnostics.Ai[:,:]
        self.Ai[  -1, 0:-1] = -self.diagnostics.Ai[0,:]
        self.Ai[   :,   -1] = self.Ai[:,0]
        
        self.J[0:-1, 0:-1] = self.diagnostics.J[:,:]
        self.J[  -1, 0:-1] = self.diagnostics.J[0,:]
        self.J[   :,   -1] = self.J[:,0]
        
        self.PB[0:-1, 0:-1] = self.diagnostics.e_magnetic[:,:]
        self.PB[  -1, 0:-1] = self.diagnostics.e_magnetic[0,:]
        self.PB[   :,   -1] = self.PB[:,0]
        
    
    def update_boundaries(self):
        
        Amin = min(self.diagnostics.A.min(), -self.diagnostics.A.max())
        Amax = max(self.diagnostics.A.max(), -self.diagnostics.A.min())
        Adif = Amax - Amin
        
        self.ATicks = np.linspace(Amin + 0.01 * Adif, Amax - 0.01 * Adif, 31)
        self.ANorm  = colors.Normalize(vmin=Amin + 0.01 * Adif, vmax=Amax - 0.01 * Adif)
        
        if self.diagnostics.inertial_mhd:
            Aimin = min(self.diagnostics.Ai.min(), -self.diagnostics.Ai.max())
            Aimax = max(self.diagnostics.Ai.max(), -self.diagnostics.Ai.min())
            Aidif = Aimax - Aimin
            
            self.AiTicks = np.linspace(Aimin + 0.01 * Aidif, Aimax - 0.01 * Aidif, 31)
            self.AiNorm  = colors.Normalize(vmin=Aimin + 0.01 * Aidif, vmax=Aimax - 0.01 * Aidif)

        Jmin = min(self.diagnostics.J.min(), -self.diagnostics.J.max())
        Jmax = min(self.diagnostics.J.max(), -self.diagnostics.J.min())
        Jdiff = (Jmax - Jmin)
        
        if Jmin == Jmax:
            Jmin -= 1.
            Jmax += 1.
        
        self.Jnorm = colors.Normalize(vmin=Jmin - 0.2*Jdiff, vmax=Jmax + 0.2*Jdiff)
        self.JTicks = np.linspace(Jmin - 0.2*Jdiff, Jmax + 0.2*Jdiff, 51, endpoint=True)
        
        
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
        Adiff = Amax - Amin
        
        self.Anorm = colors.Normalize(vmin=Amin - 0.2*Adiff, vmax=Amax + 0.2*Adiff)
        self.ATicks = np.linspace(Amin + 0.01 * Adiff, Amax - 0.01 * Adiff, 51, endpoint=True)
    
    
        Ai = -self.diagnostics.Ai
    
        Aimin = min(Ai.min(), -Ai.max())
        Aimax = max(Ai.max(), -Ai.min())
        Aidiff = Aimax - Aimin
        
        self.Ainorm = colors.Normalize(vmin=Aimin - 0.01*Aidiff, vmax=Aimax + 0.01*Aidiff)
        self.AiTicks = np.linspace(Aimin + 0.01 * Aidiff, Aimax - 0.01 * Aidiff, 51, endpoint=True)
    
    
    def update(self):
        
        if not (self.iTime == 0 or (self.iTime) % self.nPlot == 0 or self.iTime == self.ntMax):
            return
        
        self.read_data()

#        for coll in self.conts.collections:
#            self.axes.collections.remove(coll)
         
#        self.conts = self.axes.contourf(self.x, self.y, self.Ai.T, 51, cmap=plt.get_cmap('Paired'))
        
        self.pcm_J.set_array(self.Ai.T.ravel())
        
        plt.draw()
        
        filename = self.prefix + str('_X_%06d' % self.iTime) + '.png'
        plt.savefig(filename, dpi=100)
#        filename = self.prefix + str('_X_%06d' % self.iTime) + '.pdf'
#        plt.savefig(filename)
    
    
    def add_timepoint(self):
        self.iTime += 1
#         self.title.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        
    


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
    parser = argparse.ArgumentParser(description='Inertial MHD Solver in 2D')
    
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
    
