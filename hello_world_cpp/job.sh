#!/bin/bash
#PBS -N my_mpi_job
#PBS -A ntnu605
#PBS -l walltime=00:01:00
#PBS -l select=1:ncpus=32:mpiprocs=16
 
module load mpt
module load gcc/4.8.2 intelcomp/14.0.1 boost/1.55.0
 
cd $PBS_O_WORKDIR
 
mpiexec_mpt /work/permve/permve-ntnu-tdt4200/hello_world_cpp/hello_world

