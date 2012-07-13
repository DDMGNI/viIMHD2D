
import numpy as np

A0 = 1E-3
R0 = 0.3
u0 = 0


#nx = 128
#ny = 64

nx = 64
ny = 32

x1 = -1.0
x2 = +1.0

y1 = -0.5
y2 = +0.5


hx = (x2-x1) / (nx+1) 
hy = (y2-y1) / (ny+1)



def magnetic_potential(x, y):
    r = np.sqrt(x**2 + y**2)
    
    if r <= R0:
        A = A0 * (R0 - r)
    else:
        A = 0.
     
    return A


def magnetic_x(x, y, Lx, Ly):
    A1 = magnetic_potential(x, y-hy)
    A2 = magnetic_potential(x, y+hy)
    
    return (A2 - A1) / hy


def magnetic_y(x, y, Lx, Ly):
    A1 = magnetic_potential(x-hx, y)
    A2 = magnetic_potential(x+hx, y)
    
    return - (A2 - A1) / hx


#def magnetic_x(x, y, Lx, Ly):
#    if x == 0 and y == 0:
#        return 0.
#    else:
#        return - A0 * y / np.sqrt(x**2 + y**2)
#
#def magnetic_y(x, y, Lx, Ly):
#    if x == 0 and y == 0:
#        return 0.
#    else:
#        return + A0 * x / np.sqrt(x**2 + y**2)

def velocity_x(x, y, Lx, Ly):
    return u0

def velocity_y(x, y, Lx, Ly):
    return u0
