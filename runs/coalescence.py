
import numpy as np

u0 = 0.001
A0 = 0.05
B0 = 0.2
p0 = 1.0


def magnetic_x(x, y, Lx, Ly):
    return - B0 * np.pi * np.sin(4. * np.pi * y)
    
def magnetic_y(x, y, Lx, Ly):
    return - B0 * np.pi * np.sin(4. * np.pi * x)
    
def velocity_x(x, y, Lx, Ly):
    return + u0 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    
def velocity_y(x, y, Lx, Ly):
    return - u0 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    
def vector_potential(x, y, Lx, Ly):
    return A0 * (np.cos(4 * np.pi * x) - np.cos(4 * np.pi * y))
    
def pressure(x, y, Lx, Ly):
    return p0 + 8. * np.pi**2 * vector_potential(x, y, Lx, Ly)**2
    
