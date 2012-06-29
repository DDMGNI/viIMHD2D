'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time

import Config

from PETSc_MHD_VI  import PETScSolver
from PETSc_MHD_RK4 import PETScRK4



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
        Ly   = cfg['grid']['Ly']                    # spatial domain in y
        
        self.hx = Lx / nx                           # gridstep size in x
        self.hy = Ly / ny                           # gridstep size in y
        
        
        self.time = PETSc.Vec().createMPI(1, 1, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_rtol', cfg['solver']['petsc_residual'])
        OptDB.setValue('ksp_max_it', 100)

#        OptDB.setValue('ksp_monitor', '')
#        OptDB.setValue('log_info', '')
#        OptDB.setValue('log_summary', '')
        
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[nx, ny],
                                    proc_sizes=[PETSc.PETSC_DECIDE, PETSc.PETSC_DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=1,
                                    stencil_type='box')
        
        
        # create DA (dof = number of species + 1 for the potential)
        self.da6 = PETSc.DA().create(dim=2, dof=6,
                                     sizes=[nx, ny],
                                     proc_sizes=[PETSc.PETSC_DECIDE, PETSc.PETSC_DECIDE],
                                     boundary_type=('periodic', 'periodic'),
                                     stencil_width=1,
                                     stencil_type='box')
        
        
        # create DA for Poisson guess
        self.dax = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[nx],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                    boundary_type=('none'))
        
        # create DA for y grid
        self.day = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[ny],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                    boundary_type=('none'))
        
        
        # initialise grid
        self.da1.setUniformCoordinates(xmin=0.0, xmax=Lx*(nx-1.)/nx, 
                                       ymin=0.0, ymax=Ly*(ny-1.)/ny)
        
        self.da6.setUniformCoordinates(xmin=0.0, xmax=Lx*(nx-1.)/nx, 
                                       ymin=0.0, ymax=Ly*(ny-1.)/ny)
        
        self.dax.setUniformCoordinates(xmin=0.0, xmax=Lx*(nx-1.)/nx) 
        
        self.day.setUniformCoordinates(xmin=0.0, xmax=Ly*(ny-1.)/ny)
        
        
        # create solution and RHS vector
        self.x  = self.da6.createGlobalVec()
        self.b  = self.da6.createGlobalVec()
        
        # create vectors for magnetic and velocity field
        self.Bx = self.da1.createGlobalVec()
        self.By = self.da1.createGlobalVec()
        self.Vx = self.da1.createGlobalVec()
        self.Vy = self.da1.createGlobalVec()
        
        # set variable names
        self.x.setName('solver_x')
        self.b.setName('solver_b')
        
        self.Bx.setName('Bx')
        self.By.setName('By')
        self.Vx.setName('Vx')
        self.Vy.setName('Vy')
        
        
        # create Matrix object
#        _solver = __import__(cfg['solver']['solver_module'], fromlist=['PETScSolver'])
#        self.vp = _solver.Solver(self.grid, cfg['solver']['solver_method'])
        
#        self.petsc_mat = _solver.PETScSolver(self.da, self.x, self.b,
        self.petsc_mat = PETScSolver(self.da1, self.da6,
                                     self.x, self.b, 
                                     nx, ny, self.ht, self.hx, self.hy)
        
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
        
        
        # create Arakawa solver object
        self.mhd_rk4 = PETSc_MHD_RK4(self.da1, self.da6, nx, ny, self.ht, self.hx, self.hy)
        
        
        # set initial data
        (xs, xe), (ys, ye) = self.da6.getRanges()
        
        coords  = self.da6.getVecArray(self.da6.getCoordinates())
        
        x_arr  = self.da6.getVecArray(self.x)
        Bx_arr = self.da1.getVecArray(self.Bx)
        By_arr = self.da1.getVecArray(self.By)
        Vx_arr = self.da1.getVecArray(self.Vx)
        Vy_arr = self.da1.getVecArray(self.Vy)
        
        
        if cfg['initial_data']['magnetic_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['magnetic_python'], globals(), locals(), ['magnetic'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    Bx_arr[i,j] = init_data.velocity_x(coords[i,j][0], coords[i,j][1], Lx, Ly) 
                    By_arr[i,j] = init_data.velocity_y(coords[i,j][0], coords[i,j][1], Lx, Ly) 
        
        else:
            Bx_arr[xs:xe] = cfg['initial_data']['magnetic']            
            By_arr[xs:xe] = cfg['initial_data']['magnetic']            
            
            
        if cfg['initial_data']['velocity_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['velocity_python'], globals(), locals(), ['velocity'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    Vx_arr[i,j] = init_data.velocity_x(coords[i,j][0], coords[i,j][1], Lx, Ly) 
                    Vy_arr[i,j] = init_data.velocity_y(coords[i,j][0], coords[i,j][1], Lx, Ly) 
        
        else:
            Vx_arr[xs:xe] = cfg['initial_data']['velocity']            
            Vy_arr[xs:xe] = cfg['initial_data']['velocity']            
            
        
        # copy distribution function to solution vector
        x_arr[xs:xe, ys:ye, 0] = Bx_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 1] = By_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 2] = Vx_arr[xs:xe, ys:ye]
        x_arr[xs:xe, ys:ye, 3] = Vy_arr[xs:xe, ys:ye]
        
        
        # update solution history
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
        
        self.hdf5_viewer(self.x)
        self.hdf5_viewer(self.b)
        
        self.hdf5_viewer(self.Bx)
        self.hdf5_viewer(self.By)
        self.hdf5_viewer(self.Vx)
        self.hdf5_viewer(self.Vy)
        
        
    
    def __del__(self):
        del self.hdf5_viewer
        
    
    
    def run(self):
        
        (xs, xe), (ys, ye) = self.da6.getRanges()
            
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            
            # calculate initial guess for distribution function
            self.mhd_rk4.rk4(self.x)
            
            
            # build RHS and solve
            self.petsc_mat.formRHS(self.b)
            self.ksp.solve(self.b, self.x)
            
            
            # copy solution to B and V vectors
            x_arr  = self.da6.getVecArray(self.x)
            Bx_arr = self.da1.getVecArray(self.Bx)
            By_arr = self.da1.getVecArray(self.By)
            Vx_arr = self.da1.getVecArray(self.Vx)
            Vy_arr = self.da1.getVecArray(self.Vy)

            Bx_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 0]
            By_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 1]
            Vx_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 2]
            Vy_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 3]
            
            
            # save to hdf5 file
#            if itime % self.nsave == 0 or itime == self.grid.nt + 1:
            self.hdf5_viewer.HDF5SetTimestep(itime)
            self.hdf5_viewer(self.time)
            
            self.hdf5_viewer(self.x)
            self.hdf5_viewer(self.b)
            
            self.hdf5_viewer(self.Bx)
            self.hdf5_viewer(self.By)
            self.hdf5_viewer(self.Vx)
            self.hdf5_viewer(self.Vy)
            
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("   %5i iterations,   residual = %24.16E " % (self.ksp.getIterationNumber(), self.ksp.getResidualNorm()) )
            
            self.petsc_mat.update_history(self.x)
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc MHD Solver in 2D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscMHD2D(args.runfile)
    petscvp.run()
    
