'''
Created on Jul 02, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter

from imhd.plots.movie_plot_base import PlotMHD2DbaseMovie

class PlotMHD2D(PlotMHD2DbaseMovie):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, nTime=0, ntMax=0, nPlot=1, write=False):
        '''
        Constructor
        '''
        
        super().__init__(diagnostics, nTime=nTime, nPlot=nPlot, ntMax=ntMax, write=write)
        
        self.axes["M"].set_title('$A (x,y)$')
        
    
    def update_sub(self, iTime):
        '''
        Update plot.
        '''
        
        super().update_sub(iTime)
        
#         self.conts["M"] = self.axes["M"].contour(self.x, self.y, self.Ai.T, self.ATicks, extend='neither')
        
        self.axes["M"].cla()
        self.axes["M"].pcolormesh(self.x, self.y, self.Ai.T, norm=self.AiNorm)
        self.axes["M"].set_xlim((self.x[0], self.x[-1]))
        self.axes["M"].set_ylim((self.y[0], self.y[-1])) 
