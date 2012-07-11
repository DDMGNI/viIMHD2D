'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time

from config import Config

#from PETSc_MHD_VI  import PETScSolver
#from PETSc_MHD_VI_Simple import PETScSolver
from PETSc_MHD_VI_CFD_Simple import PETScSolver
#from PETSc_MHD_VI_DF_Simple import PETScSolver
#from PETSc_MHD_VI_NL_Simple import PETScSolver

from PETSc_MHD_Poisson_CFD import PETScPoissonSolver

#from PETSc_MHD_RK4 import PETScRK4



class petscMHD2D(object):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # load run config file
        cfg = Config(cfgfile)
        
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
        
        
        self.hx = Lx / (nx+1)                       # gridstep size in x
        self.hy = Ly / (ny+1)                       # gridstep size in y
        
        
        self.time = PETSc.Vec().createMPI(1, PETSc.DECIDE, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_rtol', cfg['solver']['petsc_residual'])
#        OptDB.setValue('ksp_max_it', 100)
        OptDB.setValue('ksp_max_it', 200)
#        OptDB.setValue('ksp_max_it', 1000)

#        OptDB.setValue('ksp_monitor', '')
#        OptDB.setValue('log_info', '')
#        OptDB.setValue('log_summary', '')
        
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[nx, ny],
                                    proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=1,
                                    stencil_type='box')
        
        
        # create DA (dof = 5 for Bx, By, Vx, Vy, P)
        self.da4 = PETSc.DA().create(dim=2, dof=7,
                                     sizes=[nx, ny],
                                     proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=1,
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
        
        self.dax.setUniformCoordinates(xmin=x1, xmax=x2) 
        
        self.day.setUniformCoordinates(xmin=y1, xmax=y2)
        
        
        # create solution and RHS vector
        self.x  = self.da4.createGlobalVec()
        self.b  = self.da4.createGlobalVec()
#        self.U  = self.da4.createGlobalVec()
        
        # create vectors for magnetic and velocity field
        self.Bx = self.da1.createGlobalVec()
        self.By = self.da1.createGlobalVec()
        self.Vx = self.da1.createGlobalVec()
        self.Vy = self.da1.createGlobalVec()
        self.P  = self.da1.createGlobalVec()
        self.Ux = self.da1.createGlobalVec()
        self.Uy = self.da1.createGlobalVec()
        self.Pb = self.da1.createGlobalVec()
        
        # set variable names
        self.x.setName('solver_x')
        self.b.setName('solver_b')
        
        self.Bx.setName('Bx')
        self.By.setName('By')
        self.Vx.setName('Vx')
        self.Vy.setName('Vy')
        self.P.setName('P')
        self.Ux.setName('Ux')
        self.Uy.setName('Uy')
        
        
        # create Matrix object
#        _solver = __import__(cfg['solver']['solver_module'], fromlist=['PETScSolver'])
#        self.vp = _solver.Solver(self.grid, cfg['solver']['solver_method'])
        
#        self.petsc_mat = _solver.PETScSolver(self.da, self.x, self.b,
        self.petsc_mat = PETScSolver(self.da4, nx, ny, self.ht, self.hx, self.hy)
        
        # create sparse matrix
        self.A = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], comm=PETSc.COMM_WORLD)
        self.A.setPythonContext(self.petsc_mat)
        self.A.setUp()
        
        # create linear solver and preconditioner
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.A)
        self.ksp.setType(cfg['solver']['petsc_ksp_type'])
        self.ksp.setInitialGuessNonzero(True)
        
        self.pc = self.ksp.getPC()
        self.pc.setType(cfg['solver']['petsc_pc_type'])
        
        
        # create Poisson matrix and solver
        self.poisson_mat = PETScPoissonSolver(self.da1, self.da4, self.x, 
                                              nx, ny, self.ht, self.hx, self.hy)
        
        self.pA = PETSc.Mat().createPython([self.P.getSizes(), self.Pb.getSizes()], comm=PETSc.COMM_WORLD)
        self.pA.setPythonContext(self.poisson_mat)
        self.pA.setUp()
        
        self.pksp = PETSc.KSP().create()
        self.pksp.setFromOptions()
        self.pksp.setOperators(self.pA)
        self.pksp.setType(cfg['solver']['petsc_ksp_type'])
        self.pksp.setInitialGuessNonzero(True)
        
        self.ppc = self.pksp.getPC()
        self.ppc.setType('none')
        
        
        # create Arakawa solver object
#        self.mhd_rk4 = PETScRK4(self.da4, nx, ny, self.ht, self.hx, self.hy)
        
        
        # set initial data
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        coords = self.da1.getCoordinateDA().getVecArray(self.da1.getCoordinates())
        
        x_arr  = self.da4.getVecArray(self.x)
        Bx_arr = self.da1.getVecArray(self.Bx)
        By_arr = self.da1.getVecArray(self.By)
        Vx_arr = self.da1.getVecArray(self.Vx)
        Vy_arr = self.da1.getVecArray(self.Vy)
        P_arr  = self.da1.getVecArray(self.P)
        Ux_arr = self.da1.getVecArray(self.Ux)
        Uy_arr = self.da1.getVecArray(self.Uy)
        
        
        if cfg['initial_data']['magnetic_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['magnetic_python'], globals(), locals(), ['magnetic_x', 'magnetic_y'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    Bx_arr[i,j] = init_data.magnetic_x(coords[i,j][0], coords[i,j][1], Lx, Ly) 
                    By_arr[i,j] = init_data.magnetic_y(coords[i,j][0], coords[i,j][1], Lx, Ly) 
        
        else:
            Bx_arr[xs:xe, ys:ye] = cfg['initial_data']['magnetic']            
            By_arr[xs:xe, ys:ye] = cfg['initial_data']['magnetic']            
            
            
        if cfg['initial_data']['velocity_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['velocity_python'], globals(), locals(), ['velocity_x', 'velocity_y'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    Vx_arr[i,j] = init_data.velocity_x(coords[i,j][0], coords[i,j][1], Lx, Ly) 
                    Vy_arr[i,j] = init_data.velocity_y(coords[i,j][0], coords[i,j][1], Lx, Ly) 
        
        else:
            Vx_arr[xs:xe, ys:ye] = cfg['initial_data']['velocity']            
            Vy_arr[xs:xe, ys:ye] = cfg['initial_data']['velocity']            
            
        Ux_arr[xs:xe, ys:ye] = Vx_arr[xs:xe, ys:ye]
        Uy_arr[xs:xe, ys:ye] = Vy_arr[xs:xe, ys:ye]
        
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                P_arr[i,j] = 0.1 + 0.5 * (Bx_arr[i,j]**2 + By_arr[i,j]**2)
        
        
        # copy distribution function to solution vector
        x_arr[xs:xe, ys:ye, 0] = Bx_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 1] = By_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 2] = Vx_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 3] = Vy_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 4] = P_arr [xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 5] = Ux_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 6] = Uy_arr[xs:xe, ys:ye]
        
        
        # update solution history
        self.petsc_mat.update_history(self.x)
        self.petsc_mat.update_history(self.x)
        
        
        # create HDF5 output file
        self.hdf5_viewer = PETSc.Viewer().createHDF5(cfg['io']['hdf5_output'],
                                          mode=PETSc.Viewer.Mode.WRITE,
                                          comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.HDF5PushGroup("/")
        
        
        # write grid data to hdf5 file
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
        
        coords_x.setName('x')
        coords_y.setName('y')
        
        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_y)
        
        
        # write initial data to hdf5 file
        self.hdf5_viewer.HDF5SetTimestep(0)
        self.hdf5_viewer(self.time)
        
#        self.hdf5_viewer(self.x)
#        self.hdf5_viewer(self.b)
        
        self.hdf5_viewer(self.Bx)
        self.hdf5_viewer(self.By)
        self.hdf5_viewer(self.Vx)
        self.hdf5_viewer(self.Vy)
        self.hdf5_viewer(self.P)
        self.hdf5_viewer(self.Ux)
        self.hdf5_viewer(self.Uy)
        
        
    
    def __del__(self):
#        if self.hdf5_viewer != None:
#            del self.hdf5_viewer
        pass
        
    
    
    def run(self):
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
            
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            # calculate initial guess
            self.calculate_initial_guess()
            
            # build RHS and solve
#            self.petsc_mat.formRHS(self.b)
#            self.ksp.solve(self.b, self.x)
            
            # update history
            self.petsc_mat.update_history(self.x)
            
            # copy solution to B and V vectors
            x_arr  = self.da4.getVecArray(self.x)
            Bx_arr = self.da1.getVecArray(self.Bx)
            By_arr = self.da1.getVecArray(self.By)
            Vx_arr = self.da1.getVecArray(self.Vx)
            Vy_arr = self.da1.getVecArray(self.Vy)
            P_arr  = self.da1.getVecArray(self.P)
            Ux_arr = self.da1.getVecArray(self.Ux)
            Uy_arr = self.da1.getVecArray(self.Uy)

            Bx_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 0]
            By_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 1]
            Vx_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 2]
            Vy_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 3]
            P_arr [xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 4]
            Ux_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 5]
            Uy_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 6]
            
            
            # save to hdf5 file
#            if itime % self.nsave == 0 or itime == self.grid.nt + 1:
            self.hdf5_viewer.HDF5SetTimestep(itime)
            self.hdf5_viewer(self.time)
            
#            self.hdf5_viewer(self.x)
#            self.hdf5_viewer(self.b)
            
            self.hdf5_viewer(self.Bx)
            self.hdf5_viewer(self.By)
            self.hdf5_viewer(self.Vx)
            self.hdf5_viewer(self.Vy)
            self.hdf5_viewer(self.P)
            self.hdf5_viewer(self.Ux)
            self.hdf5_viewer(self.Uy)
            
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("   Solver:  %5i iterations,   residual = %24.16E " % (self.ksp.getIterationNumber(), self.ksp.getResidualNorm()) )
           
           
           
    def calculate_initial_guess(self):
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        # solve for Bx, By, Vx, Vy
        self.petsc_mat.rk4(self.x)
        
        # calculate initial guess for total pressure
        self.poisson_mat.formRHS(self.Pb)
        self.pksp.solve(self.Pb, self.P)
        
        P_arr = self.da1.getVecArray(self.P)
        x_arr = self.da4.getVecArray(self.x)
        
        x_arr[xs:xe, ys:ye, 4] = P_arr[xs:xe, ys:ye]
        

#        # solve for Ux, Uy
#        x_arr = self.da4.getVecArray(self.x)
#        x_arr[xs:xe, ys:ye, 5] = x_arr[xs:xe, ys:ye, 2]
#        x_arr[xs:xe, ys:ye, 6] = x_arr[xs:xe, ys:ye, 3]
#        
#        self.petsc_mat.rk4(self.x, solveU=True)
#        
#        # calculate initial guess for total pressure
#        self.poisson_mat.formRHS(self.Pb)
#        self.pksp.solve(self.Pb, self.P)
#        
#        P_arr = self.da1.getVecArray(self.P)
#        x_arr = self.da4.getVecArray(self.x)
#        
#        x_arr[xs:xe, ys:ye, 4] = P_arr[xs:xe, ys:ye]
#        
#        # solve for Bx, By, Vx, Vy
#        self.petsc_mat.rk4(self.x)
            
        if PETSc.COMM_WORLD.getRank() == 0:
            print("   Poisson: %5i iterations,   residual = %24.16E " % (self.pksp.getIterationNumber(), self.pksp.getResidualNorm()) )
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc MHD Solver in 2D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscMHD2D(args.runfile)
    petscvp.run()
    
