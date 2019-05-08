#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)
export ESSL_DIR=${ESSL_DIR:="${OLCF_ESSL_ROOT}"}
export OLCF_MAGMA_ROOT=${OLCF_MAGMA_ROOT:="${MEMBERWORK}/mat187/deleteme/magma-2.5.0"}
export CC=${CC:=xlC_r}
export FC=${FC:=xlf90_r}
#export FC=${FC:=mpif90}
export CXX=${CXX:=xlC_r}
export BLAS_VENDOR=${BLAS_VENDOR:=IBM}
export BML_OPENMP=${BML_OPENMP:=omp}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export MAGMA_ROOT=$OLCF_MAGMA_ROOT
export BML_MAGMA=${BML_MAGMA:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-qsmp=omp"}
export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-qxlf2003=polymorphic -qthreaded -L${ESSL_DIR}/lib64 -lessl -lesslsmp "}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-qsmp=omp"}


./build.sh configure

                                                                                                                                                                                              
                                    
