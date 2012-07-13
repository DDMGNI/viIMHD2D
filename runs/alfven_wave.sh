#!/bin/bash
#
#$ -cwd
#
#$ -j y
#
#$ -l h_cpu=24:00:00
#
#$ -pe mpich2_tok_devel 32
#
#$ -m e
#$ -M michael.kraus@ipp.mpg.de
#
#$ -notify
#
#$ -N petscMHD2D
#


RUNID=alfven_wave


module load intel/12.1
module load mkl/10.3
module load impi

module load python27/python
module load python27/numpy/1.6.1
module load mpi4py/1.3.0
module load python27/cython


export RUN_DIR=/afs/ipp/home/m/mkraus/Python/code/petscMHD2D

export PETSC_DIR=/afs/ipp/home/m/mkraus/Python/petsc-3.3-p1
export PETSC_ARCH=arch-linux2-c-debug

export PYTHONPATH=$RUN_DIR:$PYTHONPATH


cd $RUN_DIR

mpiexec -np 32 python petsc_mhd2d.py runs/$RUNID.cfg

