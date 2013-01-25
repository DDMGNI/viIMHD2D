
import numpy as np

p0 = 1.0

nx = 256
ny = 256


def magnetic_potential(x, y):
    return np.cos(2.*y) - 2. * np.cos(x)

def velocity_potential(x, y):
    return 2. * np.sin(y) - 2. * np.cos(x)


def magnetic_x(x, y, Lx, Ly):
    hy = Ly / ny

    A1 = magnetic_potential(x, y-0.5*hy)
    A2 = magnetic_potential(x, y+0.5*hy)
    
    return + (A2 - A1) / hy


def magnetic_y(x, y, Lx, Ly):
    hx = Lx / nx
    
    A1 = magnetic_potential(x-0.5*hx, y)
    A2 = magnetic_potential(x+0.5*hx, y)
    
    return - (A2 - A1) / hx


def velocity_x(x, y, Lx, Ly):
    hy = Ly / ny

    V1 = velocity_potential(x, y-0.5*hy)
    V2 = velocity_potential(x, y+0.5*hy)
    
    return + (V2 - V1) / hy


def velocity_y(x, y, Lx, Ly):
    hx = Lx / nx
    
    V1 = velocity_potential(x-0.5*hx, y)
    V2 = velocity_potential(x+0.5*hx, y)
    
    return - (V2 - V1) / hx


def pressure(x, y, Lx, Ly):
    return p0
