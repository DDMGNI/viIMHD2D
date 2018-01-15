
import numpy as np

u0 = 0.1
B0 = 1.0
c  = 10.
p0 = 0.1

x1 = 0.5
x2 = 1.5

c  = 5.

def magnetic_x(x, y, hx, hy):
    if x1 <= y and y <= x2:
        return - B0
    else:
        return + B0

def magnetic_y(x, y, hx, hy):
    return 0.0
    
def velocity_x(x, y, hx, hy):
    return 0.0

def velocity_y(x, y, hx, hy):
    return u0 * np.sin(np.pi * x)

def pressure(x, y, hx, hy):
    return p0
