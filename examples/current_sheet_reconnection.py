
import numpy as np

u0 = 1E-3
B0 = 1.29
p0 = 1.0


def magnetic_x(x, y, hx, hy):
    return 0.0

def magnetic_y(x, y, hx, hy):
    return - 2*B0 * np.tanh(x) / np.cosh(x)**2    

def velocity_x(x, y, hx, hy):
    return + u0 * ( np.sin(x+y) + np.sin(x-y) )

def velocity_y(x, y, hx, hy):
    return - u0 * ( np.sin(x+y) - np.sin(x-y) )

def pressure(x, y, hx, hy):
    return p0
