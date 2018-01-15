
import numpy as np

A0 = 1E-3
R0 = 0.3
p0 = 1.0
u0 = np.sqrt(5.)
th = np.arctan(0.5)

vx = u0 * np.cos(th)
vy = u0 * np.sin(th)


def magnetic_x(x, y, hx, hy):
    r = np.sqrt(x**2 + y**2)
    
    if r < R0:
        B = - A0 * y / r
    else:
        B = 0.
    
    return B
    

def magnetic_y(x, y, hx, hy):
    r = np.sqrt(x**2 + y**2)
    
    if r < R0 and r > 0:
        B = + A0 * x / r
    else:
        B = 0.
    
    return B


def velocity_x(x, y, hx, hy):
    return vx

def velocity_y(x, y, hx, hy):
    return vy

def pressure(x, y, hx, hy):
    return p0
