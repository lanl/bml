#!/bin/bash

# Make sure all the paths are correct

rm -r build.broadwell
rm -r install.broadwell

MY_PATH=$(pwd)

<<<<<<< HEAD
export CC=${CC:=gcc}
export FC=${FC:=gfortran}
export CXX=${CXX:=g++}
export BLAS_VENDOR=${BLAS_VENDOR:=GNU}
<<<<<<< HEAD
=======
export CC=${CC:=icc}
export FC=${FC:=ifort}
export CXX=${CXX:=icpc}
export BLAS_VENDOR=${BLAS_VENDOR:=MKL}
>>>>>>> vectorization work on bml
export BML_OPENMP=${BML_OPENMP:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install.broadwell"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-g -O3 -qopenmp-simd"}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-g -O3 -qopenmp-simd"}
=======
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
>>>>>>> vectorization work on bml

./build.sh configure                                                                                                                                                                                              
                                    
