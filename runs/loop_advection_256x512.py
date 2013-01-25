
import numpy as np

A0 = 1E-3
R0 = 0.3
p0 = 1.0
u0 = np.sqrt(5.)
th = 0.5

vx = u0 * np.cos(np.arctan(th))
vy = u0 * np.sin(np.arctan(th))

nx = 512
ny = 256

x1 = -1.0
x2 = +1.0

y1 = -0.5
y2 = +0.5


hx = (x2-x1) / nx
hy = (y2-y1) / ny


def magnetic_potential(x, y):
    r = np.sqrt(x**2 + y**2)
    
    if r < R0:
        A = A0 * (R0 - r)
    else:
        A = 0.
     
    return A


def magnetic_x(x, y, Lx, Ly):
    A1 = magnetic_potential(x, y-0.5*hy)
    A2 = magnetic_potential(x, y+0.5*hy)
    
    return + (A2 - A1) / hy

def magnetic_y(x, y, Lx, Ly):
    A1 = magnetic_potential(x-0.5*hx, y)
    A2 = magnetic_potential(x+0.5*hx, y)
    
    return - (A2 - A1) / hx

def velocity_x(x, y, Lx, Ly):
    return vx

def velocity_y(x, y, Lx, Ly):
    return vy

def pressure(x, y, Lx, Ly):
    return p0
