
import numpy as np

u0 = 1.0
B0 = 1.0
p0 = 0.1

Bx0 = B0
By0 = B0


def magnetic_x(x, y, Lx, Ly):
#    return 0.01 * Bx0 * np.cos(np.pi * y)
#    return Bx0 * np.cos(np.pi * y)
    return Bx0
#    return 0.0

def magnetic_y(x, y, Lx, Ly):
    return By0 * np.sin(np.pi * x)
#    return By0 * np.cos(np.pi * x)
#    return By0
#    return 0.0

def velocity_x(x, y, Lx, Ly):
#    return u0 * np.sin(np.pi * y)
#    return u0 * ( 1. + 0.01 * np.cos(np.pi * y) )
    return u0
#    return 0.0
#    return 4. * (x - 1.0)
#    return 4. * (y - 1.0)
#    return 1.

def velocity_y(x, y, Lx, Ly):
#    return u0 * ( 1. + 0.01 * np.cos(np.pi * x) )
    return u0 * np.cos(np.pi * x)
#    return u0 * np.sin(np.pi * x)
#    return u0
#    return 0.0
#    return 2.

def pressure(x, y, Lx, Ly):
    return p0
