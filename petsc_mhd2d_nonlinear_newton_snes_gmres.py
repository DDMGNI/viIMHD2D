'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time
import numpy as np 

from config import Config

#import PETSc_MHD_NL_Jacobian_Matrix

#from PETSc_MHD_NL_Jacobian_Matrix5d_dofs import PETScJacobian
#from PETSc_MHD_NL_Function               import PETScFunction

#from PETSc_MHD_NL_SG_Jacobian_Matrix5d import PETScJacobian
#from PETSc_MHD_NL_SG_Function          import PETScFunction

#from PETSc_MHD_NL_FV_Jacobian_Matrix5d import PETScJacobian
#from PETSc_MHD_NL_FV_Function          import PETScFunction

#from PETSc_MHD_NL_FVD_Jacobian_Matrix5d import PETScJacobian
#from PETSc_MHD_NL_FVD_Function          import PETScFunction

#from PETSc_MHD_NL_FVG_Jacobian_Matrix5d import PETScJacobian    # Gawlik Scheme
#from PETSc_MHD_NL_FVG_Function          import PETScFunction

from PETSc_MHD_NL_FVM_Jacobian_Matrix5d import PETScJacobian     # our Scheme
from PETSc_MHD_NL_FVM_Function          import PETScFunction
from PETSc_MHD_NL_FVM_Matrix            import PETScMatrix



class petscMHD2D(object):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
#        stencil = 1
        stencil = 2
        
        # load run config file
        cfg = Config(cfgfile)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
#        OptDB.setValue('snes_lag_preconditioner', 5)
        
        OptDB.setValue('snes_atol',   cfg['solver']['petsc_residual'])
        OptDB.setValue('snes_rtol',   1E-16)
        OptDB.setValue('snes_stol',   1E-18)
        OptDB.setValue('snes_max_it', 20)
        
        OptDB.setValue('ksp_rtol',   1E-8)
        OptDB.setValue('ksp_max_it',  100)
        
        OptDB.setValue('snes_monitor', '')
        OptDB.setValue('ksp_monitor', '')

        
        # timestep setup
        self.ht    = cfg['grid']['ht']              # timestep size
        self.nt    = cfg['grid']['nt']              # number of timesteps
        self.nsave = cfg['io']['nsave']             # save only every nsave'th timestep
        
        # grid setup
        nx   = cfg['grid']['nx']                    # number of points in x
        ny   = cfg['grid']['ny']                    # number of points in y
        Lx   = cfg['grid']['Lx']                    # spatial domain in x
        x1   = cfg['grid']['x1']                    # 
        x2   = cfg['grid']['x2']                    # 
        
        Ly   = cfg['grid']['Ly']                    # spatial domain in y
        y1   = cfg['grid']['y1']                    # 
        y2   = cfg['grid']['y2']                    # 
        
        self.nx = nx
        self.ny = ny
        
        if x1 != x2:
            Lx = x2-x1
        else:
            x1 = 0.0
            x2 = Lx
        
        if y1 != y2:
            Ly = y2-y1
        else:
            y1 = 0.0
            y2 = Ly
        
        
        self.hx = Lx / nx                       # gridstep size in x
        self.hy = Ly / ny                       # gridstep size in y
        
        
        # friction, viscosity and resistivity
        mu  = cfg['initial_data']['mu']                    # friction
        nu  = cfg['initial_data']['nu']                    # viscosity
        eta = cfg['initial_data']['eta']                   # resistivity
        
        
#         self.update_jacobian = True
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print()
            print("nt = %i" % (self.nt))
            print("nx = %i" % (self.nx))
            print("ny = %i" % (self.ny))
            print()
            print("ht = %e" % (self.ht))
            print("hx = %e" % (self.hx))
            print("hy = %e" % (self.hy))
            print()
            print("Lx   = %e" % (Lx))
            print("Ly   = %e" % (Ly))
            print()
            print("mu   = %e" % (mu))
            print("nu   = %e" % (nu))
            print("eta  = %e" % (eta))
            print()
        
        
        self.time = PETSc.Vec().createMPI(1, PETSc.DECIDE, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[nx, ny],
                                    proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=stencil,
                                    stencil_type='box')
        
        
        # create DA (dof = 4 for Bx, By, Vx, Vy)
        self.da4 = PETSc.DA().create(dim=2, dof=4,
                                     sizes=[nx, ny],
                                     proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=stencil,
                                     stencil_type='box')
        
        
        # create DA (dof = 5 for Bx, By, Vx, Vy, P)
        self.da5 = PETSc.DA().create(dim=2, dof=5,
                                     sizes=[nx, ny],
                                     proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=stencil,
                                     stencil_type='box')
        
        
        # create DA for x grid
        self.dax = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[nx],
                                    proc_sizes=[PETSc.DECIDE],
                                    boundary_type=('periodic'))
        
        # create DA for y grid
        self.day = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[ny],
                                    proc_sizes=[PETSc.DECIDE],
                                    boundary_type=('periodic'))
        
        
        # initialise grid
        self.da1.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.da4.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.da5.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.dax.setUniformCoordinates(xmin=x1, xmax=x2)
        
        self.day.setUniformCoordinates(xmin=y1, xmax=y2)
        
        
        # create solution and RHS vector
        self.f  = self.da5.createGlobalVec()
        self.x  = self.da5.createGlobalVec()
        self.b  = self.da5.createGlobalVec()
        
        # create global RK4 vectors
        self.Y  = self.da5.createGlobalVec()
        self.X0 = self.da5.createGlobalVec()
        self.X1 = self.da5.createGlobalVec()
        self.X2 = self.da5.createGlobalVec()
        self.X3 = self.da5.createGlobalVec()
        self.X4 = self.da5.createGlobalVec()
        
        # create local RK4 vectors
        self.localX0 = self.da5.createLocalVec()
        self.localX1 = self.da5.createLocalVec()
        self.localX2 = self.da5.createLocalVec()
        self.localX3 = self.da5.createLocalVec()
        self.localX4 = self.da5.createLocalVec()
#        self.localP  = self.da1.createLocalVec()
        
        # create vectors for magnetic and velocity field
        self.Bx = self.da1.createGlobalVec()
        self.By = self.da1.createGlobalVec()
        self.Vx = self.da1.createGlobalVec()
        self.Vy = self.da1.createGlobalVec()
        self.P  = self.da1.createGlobalVec()

        self.xcoords = self.da1.createGlobalVec()
        self.ycoords = self.da1.createGlobalVec()
        
        # create local vectors for initialisation of pressure
        self.localBx = self.da1.createLocalVec()
        self.localBy = self.da1.createLocalVec()
        self.localVx = self.da1.createLocalVec()
        self.localVy = self.da1.createLocalVec()
        
        # set variable names
        self.Bx.setName('Bx')
        self.By.setName('By')
        self.Vx.setName('Vx')
        self.Vy.setName('Vy')
        self.P.setName('P')
        
        
        # create Matrix object
        self.petsc_matrix   = PETScMatrix  (self.da1, self.da5, nx, ny, self.ht, self.hx, self.hy, mu, nu, eta)
        self.petsc_jacobian = PETScJacobian(self.da1, self.da5, nx, ny, self.ht, self.hx, self.hy, mu, nu, eta)
        self.petsc_function = PETScFunction(self.da1, self.da5, nx, ny, self.ht, self.hx, self.hy, mu, nu, eta)
        
#        self.petsc_jacobian_4d = PETSc_MHD_NL_Jacobian_Matrix.PETScJacobian(self.da1, self.da5, nx, ny, self.ht, self.hx, self.hy)
        
        
        # initialise matrix
        self.A = self.da5.createMat()
        self.A.setOption(self.A.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.A.setUp()

        # create jacobian
        self.J = self.da5.createMat()
        self.J.setOption(self.J.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.J.setUp()
        
        
        # create nonlinear solver
        self.snes = PETSc.SNES().create()
        self.snes.setFunction(self.petsc_function.snes_mult, self.f)
        self.snes.setJacobian(self.updateJacobian, self.J)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('gmres')
#         self.snes.getKSP().setType('preonly')
#         self.snes.getKSP().getPC().setType('none')
        self.snes.getKSP().getPC().setType('lu')
#        self.snes.getKSP().getPC().setFactorSolverPackage('superlu_dist')
        self.snes.getKSP().getPC().setFactorSolverPackage('mumps')
        self.snes.getKSP().getPC().setReusePreconditioner(True)
        
        
        self.ksp = None
        
        # set initial data
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
#        coords = self.da1.getCoordinatesLocal()
        
        xc_arr = self.da1.getVecArray(self.xcoords)
        yc_arr = self.da1.getVecArray(self.ycoords)
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                xc_arr[i,j] = x1 + i*self.hx
                yc_arr[i,j] = y1 + j*self.hy
        
        
        Bx_arr = self.da1.getVecArray(self.Bx)
        By_arr = self.da1.getVecArray(self.By)
        Vx_arr = self.da1.getVecArray(self.Vx)
        Vy_arr = self.da1.getVecArray(self.Vy)
        
        xc_arr = self.da1.getVecArray(self.xcoords)
        yc_arr = self.da1.getVecArray(self.ycoords)
        
        if cfg['initial_data']['magnetic_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['magnetic_python'], globals(), locals(), ['magnetic_x', 'magnetic_y'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    Bx_arr[i,j] = init_data.magnetic_x(xc_arr[i,j], yc_arr[i,j] + 0.5 * self.hy, Lx, Ly) 
                    By_arr[i,j] = init_data.magnetic_y(xc_arr[i,j] + 0.5 * self.hx, yc_arr[i,j], Lx, Ly) 
        
        else:
            Bx_arr[xs:xe, ys:ye] = cfg['initial_data']['magnetic']            
            By_arr[xs:xe, ys:ye] = cfg['initial_data']['magnetic']            
            
            
        if cfg['initial_data']['velocity_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['velocity_python'], globals(), locals(), ['velocity_x', 'velocity_y'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    Vx_arr[i,j] = init_data.velocity_x(xc_arr[i,j], yc_arr[i,j] + 0.5 * self.hy, Lx, Ly) 
                    Vy_arr[i,j] = init_data.velocity_y(xc_arr[i,j] + 0.5 * self.hx, yc_arr[i,j], Lx, Ly) 
        
        else:
            Vx_arr[xs:xe, ys:ye] = cfg['initial_data']['velocity']            
            Vy_arr[xs:xe, ys:ye] = cfg['initial_data']['velocity']            
            
        
        if cfg['initial_data']['pressure_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['pressure_python'], globals(), locals(), ['pressure', ''], 0)
        
        
        x_arr = self.da5.getVecArray(self.x)
        x_arr[xs:xe, ys:ye, 0] = Vx_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 1] = Vy_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 2] = Bx_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 3] = By_arr[xs:xe, ys:ye]
        
        
        self.da1.globalToLocal(self.Bx, self.localBx)
        self.da1.globalToLocal(self.By, self.localBy)
        self.da1.globalToLocal(self.Vx, self.localVx)
        self.da1.globalToLocal(self.Vy, self.localVy)
        
        Bx_arr = self.da1.getVecArray(self.localBx)
        By_arr = self.da1.getVecArray(self.localBy)
        Vx_arr = self.da1.getVecArray(self.localVx)
        Vy_arr = self.da1.getVecArray(self.localVy)
        P_arr  = self.da1.getVecArray(self.P)
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                P_arr[i,j] = init_data.pressure(xc_arr[i,j] + 0.5 * self.hx, yc_arr[i,j] + 0.5 * self.hy, Lx, Ly)
#                P_arr[i,j] = init_data.pressure(coords[i,j][0] + 0.5 * self.hx, coords[i,j][1] + 0.5 * self.hy, Lx, Ly) \
#                           + 0.5 * (0.25 * (Bx_arr[i,j] + Bx_arr[i+1,j])**2 + 0.25 * (By_arr[i,j] + By_arr[i,j+1])**2)
#                P_arr[i,j] = init_data.pressure(coords[i,j][0] + 0.5 * self.hx, coords[i,j][1] + 0.5 * self.hy, Lx, Ly) \
#                           + 0.5 * (0.25 * (Vx_arr[i,j] + Vx_arr[i+1,j])**2 + 0.25 * (Vy_arr[i,j] + Vy_arr[i,j+1])**2)
#                P_arr[i,j] = init_data.pressure(coords[i,j][0] + 0.5 * self.hx, coords[i,j][1] + 0.5 * self.hy, Lx, Ly) \
#                           + 0.5 * (0.25 * (Vx_arr[i,j] + Vx_arr[i+1,j])**2 + 0.25 * (Vy_arr[i,j] + Vy_arr[i,j+1])**2) \
#                           - 1.0 * (0.25 * (Bx_arr[i,j] + Bx_arr[i+1,j])**2 + 0.25 * (By_arr[i,j] + By_arr[i,j+1])**2)
        
        
        # copy distribution function to solution vector
        x_arr = self.da5.getVecArray(self.x)
        x_arr[xs:xe, ys:ye, 4] = P_arr [xs:xe, ys:ye]
        
        # update solution history
        self.petsc_matrix.update_history(self.x)
        self.petsc_jacobian.update_history(self.x)
        self.petsc_function.update_history(self.x)
        
        
        # create HDF5 output file
        self.hdf5_viewer = PETSc.ViewerHDF5().create(cfg['io']['hdf5_output'],
                                          mode=PETSc.Viewer.Mode.WRITE,
                                          comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.pushGroup("/")
        
        
        # write grid data to hdf5 file
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
        
        coords_x.setName('x')
        coords_y.setName('y')
        
        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_y)
        
        
        # write initial data to hdf5 file
        self.hdf5_viewer.setTimestep(0)
        self.save_hdf5_vectors()
        
        
    
    def __del__(self):
        self.hdf5_viewer.destroy()
        self.snes.destroy()
        
        self.A.destroy()
        self.J.destroy()
        
        self.da1.destroy()
        self.da4.destroy()
        self.da5.destroy()
        self.dax.destroy()
        self.day.destroy()
        
    
    
    def updateJacobian(self, snes, X, J, P):
        self.petsc_jacobian.update_previous(X)
        self.petsc_jacobian.formMat(J)
 
#         if self.update_jacobian:
#             self.petsc_jacobian.update_previous(X)
#             self.petsc_jacobian.formMat(J)
#             self.update_jacobian = False
#         else:
#             return PETSc.Mat().Structure.SAME_PRECONDITIONER
    
    
    def run(self):
        
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                print()
                self.time.setValue(0, self.ht*itime)
            
            # calculate initial guess
            self.calculate_initial_guess()
            
            # update Jacobian
            self.update_jacobian = True
            
            # solve
            self.snes.solve(None, self.x)
                
            # output some solver info
            fnorm = self.snes.getFunction()[0].norm()
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Linear Solver:     %5i iterations" % (self.snes.getLinearSolveIterations()) )
                print("  Nonlinear Solver:  %5i iterations,   funcnorm = %24.16E" % (self.snes.getIterationNumber(), fnorm) )
                
            
            # update history
            self.petsc_matrix.update_history(self.x)
            self.petsc_jacobian.update_history(self.x)
            self.petsc_function.update_history(self.x)
            
            # save to hdf5 file
            if itime % self.nsave == 0 or itime == self.nt + 1:
                self.save_to_hdf5()
            
    
    def calculate_initial_guess(self):
        
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.A)
        self.ksp.setType('gmres')
        self.ksp.getPC().setType('none')
#         self.ksp.setType('preonly')
#         self.ksp.getPC().setType('lu')
# #        self.ksp.getPC().setFactorSolverPackage('superlu_dist')
#         self.ksp.getPC().setFactorSolverPackage('mumps')
    
        # build matrix
        self.petsc_matrix.formMat(self.A)
        
        # build RHS
        self.petsc_matrix.formRHS(self.b)
        
        # solve
        self.ksp.solve(self.b, self.x)
        
        # destroy
        self.ksp.destroy()
        
        
            
    def rk4(self, X, Y, fac=1.0):
        
        self.da5.globalToLocal(X, self.localX0)
        x0  = self.da5.getVecArray(self.localX0)[...]
        x1 = self.da5.getVecArray(self.X1)[...]
        self.petsc_function.timestep(x0, x1)
        
        self.da5.globalToLocal(self.X1, self.localX1)
        x1 = self.da5.getVecArray(self.localX1)[...]
        x2 = self.da5.getVecArray(self.X2)[...]
        self.petsc_function.timestep(x0 + 0.5 * fac * self.ht * x1, x2)
        
        self.da5.globalToLocal(self.X2, self.localX2)
        x2 = self.da5.getVecArray(self.localX2)[...]
        x3 = self.da5.getVecArray(self.X3)[...]
        self.petsc_function.timestep(x0 + 0.5 * fac * self.ht * x2, x3)
        
        self.da5.globalToLocal(self.X3, self.localX3)
        x3 = self.da5.getVecArray(self.localX3)[...]
        x4 = self.da5.getVecArray(self.X4)[...]
        self.petsc_function.timestep(x0 + 1.0 * fac * self.ht * x3, x4)
        
        y  = self.da5.getVecArray(Y)[...]
        x0 = self.da5.getVecArray(X)[...]
        x1 = self.da5.getVecArray(self.X1)[...]
        x2 = self.da5.getVecArray(self.X2)[...]
        x3 = self.da5.getVecArray(self.X3)[...]
        x4 = self.da5.getVecArray(self.X4)[...]
        
        y[:,:,:] = x0 + fac * self.ht * (x1 + 2.*x2 + 2.*x3 + x4) / 6.


    
    
    def save_to_hdf5(self):
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # copy solution to B and V vectors
        x_arr  = self.da5.getVecArray(self.x)
        Bx_arr = self.da1.getVecArray(self.Bx)
        By_arr = self.da1.getVecArray(self.By)
        Vx_arr = self.da1.getVecArray(self.Vx)
        Vy_arr = self.da1.getVecArray(self.Vy)
        P_arr  = self.da1.getVecArray(self.P)
        
        Vx_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 0]
        Vy_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 1]
        Bx_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 2]
        By_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 3]
        P_arr [xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 4]
        
        
        # save timestep
        self.hdf5_viewer.setTimestep(self.hdf5_viewer.getTimestep() + 1)
        
        # save data
        self.save_hdf5_vectors()
    
    
    def save_hdf5_vectors(self):
        self.hdf5_viewer(self.time)
        self.hdf5_viewer(self.Bx)
        self.hdf5_viewer(self.By)
        self.hdf5_viewer(self.Vx)
        self.hdf5_viewer(self.Vy)
        self.hdf5_viewer(self.P)
        
    def check_jacobian(self):
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        eps = 1.E-7
        
        # calculate initial guess
#        self.calculate_initial_guess()
        
        # update previous iteration
        self.petsc_jacobian.update_previous(self.x)
        
        # calculate jacobian
        self.petsc_jacobian.formMat(self.J)
        
        # create working vectors
        Jx  = self.da5.createGlobalVec()
        dJ  = self.da5.createGlobalVec()
        ex  = self.da5.createGlobalVec()
        dx  = self.da5.createGlobalVec()
        dF  = self.da5.createGlobalVec()
        Fxm = self.da5.createGlobalVec()
        Fxp = self.da5.createGlobalVec()
        
        
#        sx = -2
#        sx = -1
        sx =  0
#        sx = +1
#        sx = +2

#        sy = -2
#        sy = -1
        sy =  0
#        sy = +1
#        sy = +2
        
        nfield=5
        
        for ifield in range(0, nfield):
            for ix in range(xs, xe):
                for iy in range(ys, ye):
                    for tfield in range(0, nfield):
                        
                        # compute ex
                        ex_arr = self.da5.getVecArray(ex)
                        ex_arr[:] = 0.
                        ex_arr[(ix+sx) % self.nx, (iy+sy) % self.ny, ifield] = 1.
                        
                        
                        # compute J.e
                        self.J.mult(ex, dJ)
                        
                        dJ_arr = self.da5.getVecArray(dJ)
                        Jx_arr = self.da5.getVecArray(Jx)
                        Jx_arr[ix, iy, tfield] = dJ_arr[ix, iy, tfield]
                        
                        
                        # compute F(x - eps ex)
                        self.x.copy(dx)
                        dx_arr = self.da5.getVecArray(dx)
                        dx_arr[(ix+sx) % self.nx, (iy+sy) % self.ny, ifield] -= eps
                        
                        self.petsc_function.matrix_mult(dx, Fxm)
                        
                        
                        # compute F(x + eps ex)
                        self.x.copy(dx)
                        dx_arr = self.da5.getVecArray(dx)
                        dx_arr[(ix+sx) % self.nx, (iy+sy) % self.ny, ifield] += eps
                        
                        self.petsc_function.matrix_mult(dx, Fxp)
                        
                        
                        # compute dF = [F(x + eps ex) - F(x - eps ex)] / (2 eps)
                        Fxm_arr = self.da5.getVecArray(Fxm)
                        Fxp_arr = self.da5.getVecArray(Fxp)
                        dF_arr  = self.da5.getVecArray(dF)
                        
                        dF_arr[ix, iy, tfield] = ( Fxp_arr[ix, iy, tfield] - Fxm_arr[ix, iy, tfield] ) / (2. * eps)
                        
            
            diff = np.zeros(nfield)
            
            for tfield in range(0,nfield):
#                print()
#                print("Fields: (%5i, %5i)" % (ifield, tfield))
#                print()
                
                Jx_arr = self.da5.getVecArray(Jx)[...][:, :, tfield]
                dF_arr = self.da5.getVecArray(dF)[...][:, :, tfield]
                
                
#                print("Jacobian:")
#                print(Jx_arr)
#                print()
#                
#                print("[F(x+dx) - F(x-dx)] / [2 eps]:")
#                print(dF_arr)
#                print()
#                
#                print("Difference:")
#                print(Jx_arr - dF_arr)
#                print()
                
                
#                if ifield == 3 and tfield == 2:
#                    print("Jacobian:")
#                    print(Jx_arr)
#                    print()
#                    
#                    print("[F(x+dx) - F(x-dx)] / [2 eps]:")
#                    print(dF_arr)
#                    print()
                
                
                diff[tfield] = (Jx_arr - dF_arr).max()
            
            print()
        
            for tfield in range(0,nfield):
                print("max(difference)[fields=%i,%i] = %16.8E" % ( ifield, tfield, diff[tfield] ))
            
            print()
    


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc MHD Solver in 2D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscMHD2D(args.runfile)
    petscvp.run()
#    petscvp.check_jacobian()
    
