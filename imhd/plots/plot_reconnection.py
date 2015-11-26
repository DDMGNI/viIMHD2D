'''
Created on Jul 02, 2012

@author: mkraus
'''

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter

from .replay_plot_base import PlotMHD2DbaseReplay


class PlotMHD2D(PlotMHD2DbaseReplay):
    '''
    Replay plot for reconnection studies including the generalised magnetic induction. 
    '''

    def __init__(self, diagnostics, nTime=0, nPlot=1):
        '''
        Constructor
        '''
        
        super().__init__(diagnostics, nTime=0, nPlot=1)
        
        self.axes["M1"].set_title('$B (x,y)$')
        self.axes["M2"].set_title('$\\bar{B} (x,y)$')
        
        
    
    def update(self, iTime, final=False):
        '''
        Update plot.
        '''
        
        super().update(iTime, final=final, draw=False)
        
        self.conts["M1"] = self.axes["M1"].contour(self.x, self.y, self.A.T,  self.ATicks, extend='neither')
        self.conts["M2"] = self.axes["M2"].contour(self.x, self.y, self.Ai.T, self.ATicks, extend='neither')
        
        
        self.figure.canvas.draw()
        plt.show(block=final)
        plt.pause(1)
        
        return self.figure
