#!/bin/bash -l
# Job Name:
#SBATCH -J viIMHD2D
# Initial working directory:
#SBATCH -D /u/mkraus/Codes/viIMHD2D
# Standard output and error:
#SBATCH -o /ptmp/mkraus/viIMHD2D/reconnection_1024x512_part2.out.%j
#SBATCH -e /ptmp/mkraus/viIMHD2D/reconnection_1024x512_part2.err.%j
# Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32
#
#SBATCH --mail-type=all
#SBATCH --mail-user=michael.kraus@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:24:00

# Run the program:
srun python3.6 inertial_mhd2d_nonlinear_newton_snes_gmres.py runs_draco/reconnection_1024x512_part2_lu.cfg
