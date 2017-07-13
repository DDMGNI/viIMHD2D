#!/bin/bash
#
#$ -cwd
#
#$ -l h_rt=48:00:00
#
#$ -P  tokp
#$ -pe impi_hydra 64
#
#$ -o /tokp/scratch/mkraus/viIMHD2D/current_sheet_reconnection_1024x512.$JOB_ID.out
#$ -e /tokp/scratch/mkraus/viIMHD2D/current_sheet_reconnection_1024x512.$JOB_ID.err
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N viMHD2D
#

RUNID=current_sheet_reconnection_1024x512_part3

export MODULEPATH=/u/mkraus/Python/modules:${MODULEPATH}

module load intel/14.0
module load mkl/11.1
module load impi/4.1.3
module load hdf5-mpi/1.8.12

module load py33-python
module load py33-cython
module load py33-numpy
module load py33-scipy
module load py33-configobj
module load py33-h5py
module load py33-mpi4py
module load py33-petsc4py


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics2013/14.0/mkl/lib/intel64/libmkl_core.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics2013/14.0/mkl/lib/intel64/libmkl_intel_thread.so:$LD_PRELOAD
export LD_PRELOAD=/afs/@cell/common/soft/intel/ics2013/14.0/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD


export RUN_DIR=/u/mkraus/Codes/viIMHD2D
cd $RUN_DIR

mpiexec -perhost 16 -l -n 64 python3.3 inertial_mhd2d_nonlinear_newton_snes_gmres.py runs_tokp/$RUNID.cfg 

