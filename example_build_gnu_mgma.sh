#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)
#export OLCF_MAGMA_ROOT=${OLCF_MAGMA_ROOT:="${OLCF_MAGMA_ROOT}"}
export OLCF_MAGMA_ROOT=${OLCF_MAGMA_ROOT:="$PROJWORK/csc304/magma/magma-2.5.1-alpha1"}
export CC=${CC:=gcc}
export FC=${FC:=gfortran}
#export FC=${FC:=mpif90}
export CXX=${CXX:=g++}
export BLAS_VENDOR=${BLAS_VENDOR:=GNU}
export BML_OPENMP=${BML_OPENMP:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export MAGMA_ROOT=$OLCF_MAGMA_ROOT
export BML_MAGMA=${BML_MAGMA:=yes}
export BML_COMPLEX=${BML_COMPLEX:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-fopenmp"}
export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:=" "}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-L${OLCF_CUDA_ROOT}/lib64/ -lcublas -lcudart"}


./build.sh configure

                                                                                                                                                                                              
                                    
