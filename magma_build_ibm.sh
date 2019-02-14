#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)
#export ESSL_DIR=${ESSL_DIR:=/sw/summitdev/essl/5.5.0-20161110}
#export ESSL_DIR=${ESSL_DIR:=/sw/summitdev/essl/5.5.0-20161110}
export ESSL_DIR=${ESSL_DIR:=/sw/summitdev/essl/6.1.0/essl/6.1/}
export CC=${CC:=xlC_r}
export FC=${FC:=/sw/summitdev/xl/16.1.1-beta6/xlf/16.1.1/bin/xlf90_r}
export CXX=${CXX:=xlC_r}
#export BLAS_VENDOR=${BLAS_VENDOR:=IBM}
export BML_OPENMP=${BML_OPENMP:=no}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-qsmp=omp"}
export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-qxlf2003=polymorphic -qthreaded"}
export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-qxlf2003=polymorphic -qthreaded -L/sw/summitdev/essl/6.1.0/essl/6.1/lib64 -lessl -lesslsmp "}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-qsmp=omp"}
export BML_MAGMA=${BML_MAGMA:=yes}
export MAGMA_ROOT=${MAGMA_ROOT:=/ccs/home/cnegre/magma-2.5.0-orig/}


#CC=xlc_r FC=xlf_r BLAS_VENDOR=IBM CMAKE_BUILD_TYPE=Release BML_OPENMP=yes CMAKE_INSTALL_PREFIX=/home/smm/bml/bml/install CMAKE_Fortran_FLAGS="-qxlf2003=polymorphic -qthreaded" ./build.sh configure


./build.sh configure

                                                                                                                                                                                              
                                    
