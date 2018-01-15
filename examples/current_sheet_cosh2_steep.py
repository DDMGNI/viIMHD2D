
import numpy as np

u0 = 0.1
B0 = 1.0
p0 = 0.1


def magnetic_x(x, y, hx, hy):
    return 0.0

def magnetic_y(x, y, hx, hy):
    return B0 / np.cosh(2.*np.pi * (x-1.))**2 - 0.5 * B0
    
def velocity_x(x, y, hx, hy):
    return u0 * np.sin(np.pi * y)

def velocity_y(x, y, hx, hy):
    return 0.0

def pressure(x, y, hx, hy):
    return p0
