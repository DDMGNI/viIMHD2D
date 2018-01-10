
import numpy as np

A0 = 1E-3
#A0 = 1E-6
p0 = 1.0
u0 = np.sqrt(8.)
th = np.arctan(1.0)

vx = u0 * np.cos(th)
vy = u0 * np.sin(th)


def magnetic_potential(x, y):
    return A0 * np.exp(np.cos(np.pi*x) + np.cos(np.pi*y))

def magnetic_x(x, y, Lx, Ly):
    return - A0 * np.pi * np.sin(np.pi*y) * np.exp(np.cos(np.pi*x) + np.cos(np.pi*y))

def magnetic_y(x, y, Lx, Ly):
    return + A0 * np.pi * np.sin(np.pi*x) * np.exp(np.cos(np.pi*x) + np.cos(np.pi*y))
    
def velocity_x(x, y, Lx, Ly):
    return vx

def velocity_y(x, y, Lx, Ly):
    return vy

def pressure(x, y, Lx, Ly):
    return p0
