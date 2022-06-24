#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)

export CC=${CC:=gcc}
export FC=${FC:=gfortran}
export CXX=${CXX:=g++}
export BLAS_VENDOR=${BLAS_VENDOR:=GNU}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_MAGMA=${BML_MAGMA:=yes}
export BML_CUSOLVER=${BML_CUSOLVER:=yes}
export MAGMA_ROOT=${MAGMA_ROOT:="${HOME}/magma"}
export OLCF_MAGMA_ROOT=${OLCF_MAGMA_ROOT:="${HOME}/magma"}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
#export BML_COMPLEX=${BML_COMPLEX:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
#export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-fopenmp -lpthread "}  #-I${OLCF_OPENBLAS_ROOT}/include "}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-DMAGMA_LIBRARIES=home/finkeljo/magma/lib -L${CUDA_LIB} -L/usr/local/cuda-10.2/lib64 -I${CUDA_INCLUDE} -lcublas -lcudart -lcusolver -llapack -lblas"}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-L${CUDA_LIB} -L/usr/local/cuda-10.2/lib64 -I${CUDA_INCLUDE} -lcublas -lcudart -lcusolver -llapack -lblas"}

#export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-ffree-line-length-none -fopenmp -lpthread -L${OLCF_ESSL_ROOT}/lib64 -lesslsmp -L${OLCF_CUDA_ROOT}/lib64/ -lcublas -lcudart"}
#export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-ffree-line-length-none -fopenmp -lpthread -L${OLCF_ESSL_ROOT}/lib64 -lesslsmp -L${OLCF_CUDA_ROOT}/lib64/ -lcublas -lcudart  -DMAGMA_LIBRARIES=$(HOME)/magma/lib/libmagma.a"}


./build.sh configure

                                                                                                                                                                                              
                                    
