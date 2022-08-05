#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)

export CUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT ## cmake cant find CUDA_TOOLKIT_ROOT_DIR

export CC=${CC:=gcc}
export FC=${FC:=gfortran}
export CXX=${CXX:=g++}
export BLAS_VENDOR=${BLAS_VENDOR:=Generic}
export BML_OPENMP=${BML_OPENMP:=no}
export BML_MAGMA=${BML_MAGMA:=yes}
export BML_MPTC=${BML_MPTC:=yes}
export BML_CUSOLVER=${BML_CUSOLVER:=yes}
export MAGMA_ROOT=${MAGMA_ROOT:="${HOME}/magma"}
export OLCF_MAGMA_ROOT=${OLCF_MAGMA_ROOT:="${HOME}/magma"}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
#export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-DMAGMA_LIBRARIES=home/finkeljo/magma/lib -L${CUDA_LIB} -L/usr/local/cuda-11.7.0/lib64 -I${CUDA_INCLUDE} -lcublas -lcudart -lcusolver -llapack -lblas"}
#export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-L${CUDA_LIB} -L/usr/local/cuda-11.7.0/lib64 -I${CUDA_INCLUDE} -lcublas -lcudart -lcusolver -llapack -lblas"}



./build.sh configure

cd build 

make -j

make install
                                                                                                                                                                                              
                                    
