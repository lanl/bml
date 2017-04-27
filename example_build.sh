#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

#module load gcc/5.4.0
#module load openmpi/1.10.3-gcc_5.4.0
#module load cmake
#module load mkl

FC=gfortran CC=gcc BML_MPI=no CMAKE_BUILD_TYPE=Release \
INSTALL_DIR=$HOME/bml/install  BLAS_VENDOR=MKL \
BML_OPENMP=yes BML_TESTING=yes ./build.sh configure

                                                                                                                                                                                              
                                    
