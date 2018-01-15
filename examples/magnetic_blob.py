
import numpy as np

u0 = 1.0
B0 = 0.1
p0 = 0.1

Bx0 = B0
By0 = B0

x0  = 0.5
y0  = 1.0
sig = 0.25


def gaussian(x, x0, sigma):
    if x >= x0-sigma and x <= x+sigma:
        return sigma * ( 1. - (x-x0)**2 )
    else:
        return 0.
    
def magnetic_x(x, y, hx, hy):
    return Bx0 * ( 1. + 0.5 * gaussian(x, x0, sig) * gaussian(y, y0, sig) )

def magnetic_y(x, y, hx, hy):
    return By0 * ( 1. - 0.5 * gaussian(x, x0, sig) * gaussian(y, y0, sig) )


def velocity_x(x, y, hx, hy):
#    return u0 * np.sin(np.pi * x)
    return u0
#    return 0.0
#    return 4. * (x - 1.0)
#    return 4. * (y - 1.0)

def velocity_y(x, y, hx, hy):
#    return u0 * np.cos(np.pi * x)
#    return u0 * np.sin(np.pi * x)
#    return u0
    return 0.0

def pressure(x, y, hx, hy):
    return p0
