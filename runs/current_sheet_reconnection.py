
import numpy as np

u0 = 1E-3
B0 = 1.29
p0 = 0.1


def magnetic_x(x, y, Lx, Ly):
    return 0.0

def magnetic_y(x, y, Lx, Ly):
#     return - B0 / np.cosh(x)**2
    return - 2*B0 * np.tanh(x) / np.cosh(x)**2    

def velocity_x(x, y, Lx, Ly):
    return - u0 * ( np.sin(x+y) + np.sin(x-y) )

def velocity_y(x, y, Lx, Ly):
    return + u0 * ( np.sin(x+y) - np.sin(x-y) )

def pressure(x, y, Lx, Ly):
    return p0
