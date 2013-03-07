#!/bin/bash
#
#$ -cwd
#
#$ -j y
#
#$ -l h_cpu=48:00:00
#
#$ -pe mpich2_tok_production 8
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N petscMHD2D
#


RUNID=current_sheath_256


module load intel/13.1
module load mkl/11.0
module load impi/4.1.0

module load python32/all


export LD_PRELOAD=/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_core.so:/afs/@cell/common/soft/intel/ics13/13.0/mkl/lib/intel64/libmkl_intel_thread.so:/afs/@cell/common/soft/intel/ics13/13.0/compiler/lib/intel64/libiomp5.so


export RUN_DIR=/afs/ipp/home/m/mkraus/Codes/petscMHD2D

export PYTHONPATH=$RUN_DIR/vi:$PYTHONPATH


cd $RUN_DIR

mpiexec -np 8 python petsc_mhd2d_nonlinear_newton_direct.py runs/$RUNID.cfg
