
import numpy as np


def magnetic_x(x, y, Lx, Ly):
    return 1. + 0.5 * np.cos(0.5 * (x - 0.5 * Lx))

def magnetic_y(x, y, Lx, Ly):
    return 1. + 0.5 * np.sin(0.5 * (x - 0.5 * Ly))

def velocity_x(x, y, Lx, Ly):
    return 1. + 0.5 * np.cos(0.5 * (x - 0.5 * Lx))

def velocity_y(x, y, Lx, Ly):
    return 1. + 0.5 * np.sin(0.5 * (x - 0.5 * Ly))

