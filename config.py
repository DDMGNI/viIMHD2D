'''
Created on Mar 20, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from configobj import ConfigObj
from validate  import Validator

import os.path

from petsc4py import PETSc


class Config(ConfigObj):
    '''
    Run configuration.
    '''


    def __init__(self, infile, file_error=True):
        '''
        Constructor
        '''
        
        self.runspec = 'runspec.cfg'
        
        ConfigObj.__init__(self, infile=infile, configspec=self.runspec, file_error=file_error)
        
        self.validator = Validator()
        self.valid     = self.validate(self.validator, copy=True)
        
    
    def write_default_config(self):
        '''
        Reads default values from runspec file and creates a default
        configuration file in run.cfg.default.
        '''
        
        self.write()
    
    
    def set_petsc_options(self):
        
        OptDB = PETSc.Options()
        
        if self['solver']['petsc_snes_type'] == 'newton_basic':
            OptDB.setValue('snes_ls', 'basic')
            
        elif self['solver']['petsc_snes_type'] == 'newton_quadratic':
            OptDB.setValue('snes_ls', 'quadratic')
#         else:
#             raise Exception("Invalid SNES type.")
        
        OptDB.setValue('snes_atol',   self['solver']['petsc_snes_atol'])
        OptDB.setValue('snes_rtol',   self['solver']['petsc_snes_rtol'])
        OptDB.setValue('snes_stol',   self['solver']['petsc_snes_stol'])
        OptDB.setValue('snes_max_it', self['solver']['petsc_snes_max_iter'])
        OptDB.setValue('ksp_atol',    self['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_rtol',    self['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_max_it',  self['solver']['petsc_ksp_max_iter'])
        OptDB.setValue('ksp_type',    self['solver']['petsc_ksp_type'])
        OptDB.setValue('pc_type',     self['solver']['petsc_pc_type'])
        
        if self['solver']['petsc_snes_monitor']:
            OptDB.setValue('snes_monitor', '')
        
        if self['solver']['petsc_ksp_monitor']:
            OptDB.setValue('ksp_monitor', '')
        
        if self['solver']['petsc_pc_type'] == 'lu':
            OptDB.setValue('pc_factor_mat_solver_package', self['solver']['petsc_lu_package'])
            OptDB.setValue('pc_factor_solver_package', self['solver']['petsc_lu_package'])
            
        elif self['solver']['petsc_pc_type'] == 'asm':
            OptDB.setValue('pc_asm_type',  'restrict')
            OptDB.setValue('pc_asm_overlap', 3)
            OptDB.setValue('sub_ksp_type', 'preonly')
            OptDB.setValue('sub_pc_type', 'lu')
            OptDB.setValue('sub_pc_factor_mat_solver_package', self['solver']['petsc_lu_package'])
        
    

if __name__ == '__main__':
    '''
    Instantiates a Config object and creates a default configuration file.
    '''
    
    filename = 'run.cfg.default'
    
    if os.path.exists(filename):
        os.remove(filename)
    
    config = Config(filename, file_error=False)
    config.write_default_config()

