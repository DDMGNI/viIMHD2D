'''
Created on Jul 6, 2012

@author: mkraus
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec

from runs.testpetsc import magnetic_x, magnetic_y, velocity_x, velocity_y

from pyMHD2D_RK4 import pyMHD2D_RK4


class pyMHD2D(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        self.nt = 200
        self.np = 1
        
        self.nx = 30
        self.ny = 30
        
        self.Lx = 2.0
        self.Ly = 2.0
        
        self.ht = 0.001
        
        
        # create grid
        self.x = np.linspace(0.0, self.Lx, self.nx, endpoint=False)
        self.y = np.linspace(0.0, self.Ly, self.ny, endpoint=False)
        
        self.hx = self.x[1] - self.x[0]
        self.hy = self.y[1] - self.y[0]
        
        self.iTime = 0
        
        
        # create arrays to hold magnetic and velocity field values
        self.Bx = np.empty((self.nx, self.ny))
        self.By = np.empty((self.nx, self.ny))
        self.Vx = np.empty((self.nx, self.ny))
        self.Vy = np.empty((self.nx, self.ny))
        
        
        # initialise magnetic and velcocity fields
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                self.Bx[i,j] = magnetic_x(self.x[i], self.y[j], self.Lx, self.Ly)
                self.By[i,j] = magnetic_y(self.x[i], self.y[j], self.Lx, self.Ly)
                self.Vx[i,j] = velocity_x(self.x[i], self.y[j], self.Lx, self.Ly)
                self.Vy[i,j] = velocity_y(self.x[i], self.y[j], self.Lx, self.Ly)
        
        
        ############################################################
        # create plot
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.2, wspace=0.25)
        plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = %1.3f' % (self.iTime * self.ht), horizontalalignment='center') 
        
        # create plot containers
        self.axes  = {}
        self.conts = {}
        
        
        # create and setup subplots
        gs = gridspec.GridSpec(2, 2)
        self.gs = gs
        
        self.axes["Bx"] = plt.subplot(gs[0,0])
        self.axes["By"] = plt.subplot(gs[0,1])
        self.axes["Vx"] = plt.subplot(gs[1,0])
        self.axes["Vy"] = plt.subplot(gs[1,1])
        
        self.axes["Bx"].set_title('$B_{x} (x,y)$')
        self.axes["By"].set_title('$B_{y} (x,y)$')
        self.axes["Vx"].set_title('$V_{x} (x,y)$')
        self.axes["Vy"].set_title('$V_{y} (x,y)$')
        
        plt.setp(self.axes["Bx"].get_xticklabels(), visible=False)
        plt.setp(self.axes["By"].get_xticklabels(), visible=False)
        
        
        # setup boundaries for magnetic field
        self.Bmin = +1e40
        self.Bmax = -1e40
        
        self.Bmin = min(self.Bmin, self.Bx.min() )
        self.Bmin = min(self.Bmin, self.By.min() )
        
        self.Bmax = max(self.Bmax, self.Bx.max() )
        self.Bmax = max(self.Bmax, self.By.max() )

        dB = 0.1 * (self.Bmax - self.Bmin)
        self.Bnorm = colors.Normalize(vmin=self.Bmin-dB, vmax=self.Bmax+dB)


        # setup boundaries for velocity field
        self.Vmin = +1e40
        self.Vmax = -1e40
        
        self.Vmin = min(self.Vmin, self.Vx.min() )
        self.Vmin = min(self.Vmin, self.Vy.min() )
        
        self.Vmax = max(self.Vmax, self.Vx.max() )
        self.Vmax = max(self.Vmax, self.Vy.max() )

        dV = 0.1 * (self.Vmax - self.Vmin)
        self.Vnorm = colors.Normalize(vmin=self.Vmin-dV, vmax=self.Vmax+dV)


        # plot initial data
        self.update_plot()
        
        
        # create RK4 object
        self.mhd_rk4 = pyMHD2D_RK4(self.nx, self.ny, self.ht, self.hx, self.hy)
        
        
    def update_plot(self, final=False):
        
        if not (self.iTime == 0 or self.iTime % self.np == 0 or self.iTime == self.nt):
            return
        
        # update plot title
        self.title.set_text('t = %1.3f' % (self.iTime * self.ht))

        # clear contours
        for ckey, cont in self.conts.iteritems():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
                
        self.conts["Bx"] = self.axes["Bx"].contourf(self.x, self.y, self.Bx.T, 20, norm=self.Bnorm)
        self.conts["By"] = self.axes["By"].contourf(self.x, self.y, self.By.T, 20, norm=self.Bnorm)
        self.conts["Vx"] = self.axes["Vx"].contourf(self.x, self.y, self.Vx.T, 20, norm=self.Vnorm)
        self.conts["Vy"] = self.axes["Vy"].contourf(self.x, self.y, self.Vy.T, 20, norm=self.Vnorm)

        plt.draw()
        plt.show(block=final)
        
    
    
    def run(self):
        for self.iTime in range(1, self.nt+1):
            self.mhd_rk4.rk4(self.Bx, self.By, self.Vx, self.Vy)
            self.update_plot(self.iTime == self.nt)
    
        
        
if __name__ == '__main__':
    pymhd = pyMHD2D()
    pymhd.run()
    
