
import numpy as np

u0 = 0.1
B0 = 1.0
p0 = 0.1


def magnetic_x(x, y, Lx, Ly):
    return 0.0

def magnetic_y(x, y, Lx, Ly):
    return 1. / np.cosh(np.pi * x)**2
    
def velocity_x(x, y, Lx, Ly):
    return u0 * np.sin(np.pi * y)

def velocity_y(x, y, Lx, Ly):
    return 0.0

def pressure(x, y, Lx, Ly):
    return p0
