
import numpy as np

A0 = 1E-3
R0 = 0.3
p0 = 1.0
u0 = np.sqrt(5.)
th = np.arctan(0.5)

vx = u0 * np.cos(th)
vy = u0 * np.sin(th)


def magnetic_potential(x, y):
    r = np.sqrt(x**2 + y**2)
    
    if r < R0:
        A = A0 * (R0 - r)
    else:
        A = 0.
    
    return A


def magnetic_x(x, y, hx, hy):
    A1 = magnetic_potential(x, y-0.5*hy)
    A2 = magnetic_potential(x, y+0.5*hy)
    B  = + (A2 - A1) / hy
    
    return B
    

def magnetic_y(x, y, hx, hy):
    A1 = magnetic_potential(x-0.5*hx, y)
    A2 = magnetic_potential(x+0.5*hx, y)
    B  = - (A2 - A1) / hx
    
    return B


def velocity_x(x, y, hx, hy):
    return vx

def velocity_y(x, y, hx, hy):
    return vy

def pressure(x, y, hx, hy):
    return p0
