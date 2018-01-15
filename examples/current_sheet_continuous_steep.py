
import numpy as np

u0 = 0.1
B0 = 1.0
p0 = 0.1

x1 = 0.5
x2 = 1.5
c  = 20.


def magnetic_x(x, y, hx, hy):
    return 0.0

def magnetic_y(x, y, hx, hy):
    return np.tanh(c*(x-x1)) - np.tanh(c*(x-x2)) - 1.
        
def velocity_x(x, y, hx, hy):
    return u0 * np.sin(np.pi * y)

def velocity_y(x, y, hx, hy):
    return 0.0

def pressure(x, y, hx, hy):
    return p0
