#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)

export CC=${CC:=xlc_r}
export FC=${FC:=xlf_r}
#export CXX=${CXX:=g++}
export BLAS_VENDOR=${BLAS_VENDOR:=IBM}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_GPU=${BML_GPU:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-qoffload"}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:=""}
export ESSL_DIR=/opt/ibmmath/essl/6.1/
export CMAKE_Fortran_FLAGS="-qxlf2003=polymorphic -qthreaded -qsmp=omp -qoffload"
export CMAKE_C_FLAGS="-D_OPENMP=201511 -qthreaded -qsmp=omp -qoffload"
export CMAKE_CXX_FLAGS="-D_OPENMP=201511 -qthreaded -qsmp=omp -qoffload"

./build.sh install

                                                                                                                                                                                              
                                    
