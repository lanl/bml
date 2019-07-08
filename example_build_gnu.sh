#!/bin/bash

# Make sure all the paths are correct

module load gcc netlib-lapack

rm -r build
rm -r install

MY_PATH=$(pwd)
export ESSL_DIR=${ESSL_DIR:="${OLCF_ESSL_ROOT}"}
export CC=${CC:=gcc}
export FC=${FC:=gfortran}
export CXX=${CXX:=g++}
export BLAS_VENDOR=${BLAS_VENDOR:=GNU}
export BML_OPENMP=${BML_OPENMP:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export BML_COMPLEX=${BML_COMPLEX:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
#export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-qsmp=omp"}
#export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-qxlf2003=polymorphic -qthreaded -L${ESSL_DIR}/lib64 -lessl -lesslsmp "}
#export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-qsmp=omp"}


./build.sh configure

                                                                                                                                                                                              
                                    
