#!/bin/bash
#PBS -N my_mpi_job
#PBS -m permve
#PBS -A permve
#PBS -l walltime=00:01:00
#PBS -l select=1:ncpus=32:mpiprocs=16
 
module load mpt
 
cd $PBS_O_WORKDIR
 
mpiexec_mpt /work/permve/permve-ntnu-tdt4200/hello_world

