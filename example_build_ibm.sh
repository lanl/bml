#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)
export ESSL_DIR=${ESSL_DIR:="${OLCF_ESSL_ROOT}"}
export CC=${CC:=xlC_r}
export FC=${FC:=xlf90_r}
export CXX=${CXX:=xlC_r}
export BLAS_VENDOR=${BLAS_VENDOR:=IBM}
export BML_OPENMP=${BML_OPENMP:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-qsmp=omp"}
export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-qxlf2003=polymorphic -qthreaded"}
export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-qxlf2003=polymorphic -qthreaded -L/sw/summitdev/essl/6.1.0/essl/6.1/lib64 -lessl -lesslsmp "}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-qsmp=omp"}


./build.sh configure

                                                                                                                                                                                              
                                    
