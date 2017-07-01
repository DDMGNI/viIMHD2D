'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import numpy as np

import matplotlib.animation as animation

from imhd.diagnostics import Diagnostics 
from plot_PJ import PlotMHD2D
#from plot_chaco import PlotMHD2D


class replay(object):
    '''
    
    '''

    def __init__(self, hdf5_file, nPlot=1):
        '''
        Constructor
        '''
        
        self.diagnostics = Diagnostics(hdf5_file)
        
        self.nPlot = nPlot
        self.plot  = PlotMHD2D(self.diagnostics, self.diagnostics.nt, nPlot)
#        self.plot.configure_traits()
        
        self.init()
    
    def init(self):
        self.update(0)
    
    
    def update(self, itime, final=False):
        self.diagnostics.read_from_hdf5(itime)
        self.diagnostics.update_invariants(itime)
        
        if itime > 0:
            self.plot.add_timepoint()
        
        if (itime == 0 or itime == 1 or (itime-1) % self.nPlot == 0 or itime-1 == self.plot.nTime):
            self.diagnostics.calculate_divergence()
#            print("   min(Bx)    = %20.12E,     max(Bx)    = %20.12E" % (self.diagnostics.Bx.min(), self.diagnostics.Bx.max()))
#            print("   min(By)    = %20.12E,     max(By)    = %20.12E" % (self.diagnostics.By.min(), self.diagnostics.By.max()))
#            print("   min(Vx)    = %20.12E,     max(Vx)    = %20.12E" % (self.diagnostics.Vx.min(), self.diagnostics.Vx.max()))
#            print("   min(Vy)    = %20.12E,     max(Vy)    = %20.12E" % (self.diagnostics.Vy.min(), self.diagnostics.Vy.max()))
#            print("   min(div B) = %20.12E,     max(div B) = %20.12E" % (self.diagnostics.divB.min(), self.diagnostics.divB.max()))
#            print("   min(div V) = %20.12E,     max(div V) = %20.12E" % (self.diagnostics.divV.min(), self.diagnostics.divV.max()))
            print("   energy     = %20.12E" % self.diagnostics.energy)
            print("   helicity   = %20.12E" % self.diagnostics.helicity)
            print("   max(div B) = %20.12E" % max(-self.diagnostics.divB.min(), self.diagnostics.divB.max()))
            print("   max(div V) = %20.12E" % max(-self.diagnostics.divV.min(), self.diagnostics.divV.max()))
            print
        
        return self.plot.update(final=final)
    
    
    def run(self):
        for itime in range(1, self.diagnostics.nt+1):
            print("it = %4i" % (itime))
            self.update(itime, final=(itime == self.diagnostics.nt))
        
    
    def movie(self, outfile, fps=1):
        self.plot.nPlot = 1
        
        ani = animation.FuncAnimation(self.plot.figure, self.update, np.arange(1, self.diagnostics.nt+1), 
                                      init_func=self.init, repeat=False, blit=True)
        ani.save(outfile, fps=fps)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ideal MHD Solver in 2D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')    
    parser.add_argument('-o', metavar='<run.mp4>', type=str, default=None,
                        help='output video file')    
    
    args = parser.parse_args()
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = replay(args.hdf5_file, args.np)
    
    print
#    input('Hit any key to start replay.')
    input('Hit any key to start replay.')
    print
    
    if args.o != None:
        pyvp.movie(args.o, args.np)
    else:
        pyvp.run()
    
    print
    print("Replay finished.")
    print
    
