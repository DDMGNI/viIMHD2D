
import numpy as np

u0 = 1.0
B0 = 1.0

Bx0 = B0
By0 = B0

sigma   = 0.25
xcenter = 1.0


def magnetic_x(x, y, Lx, Ly):
    return Bx0

def magnetic_y(x, y, Lx, Ly):
    return By0 * gaussian(x)

def velocity_x(x, y, Lx, Ly):
    return 0.0

def velocity_y(x, y, Lx, Ly):
    return u0 * gaussian(x)

def gaussian(x):
    return 1. / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-xcenter)/sigma)**2 )
