#!/bin/bash
module load cmake
module load cuda
module load gcc/11.2.0
module load openblas

export CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR="/sw/summit/cuda/11.0.3"}

rm -r build
rm -r install

MY_PATH=$(pwd)

# change this to path of hypre installation.
# build hypre with: ./configure --with-cuda --without-MPI CUCC=nvcc
# using gcc-9 compilers.
HYPRE_INSTALL_PATH="/ccs/home/osei/soft/CoPA/with-hypre/hypre/src/hypre"

#get jsrun with full path
JSRUN=$(which jsrun)
echo ${JSRUN}

export CC=${CC:=gcc}
export FC=${FC:=gfortran}
export CXX=${CXX:=g++}
export BLAS_VENDOR=${BLAS_VENDOR:=OpenBLAS}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_OMP_OFFLOAD=${BML_OMP_OFFLOAD:=yes}
export BML_HYPRE=${BML_HYPRE:=yes}
export HYPRE_ROOT=${HYPRE_INSTALL_PATH}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export BML_COMPLEX=${BML_COMPLEX:=no}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-fopenmp -latomic -lm"}
export BML_CUSPARSE=${BML_CUSPARSE:=no}
export BML_COMPLEX=${BML_COMPLEX:=no}
export BML_SYEVD=${BML_SYEVD:=no}

#use jsrun to run tests on a compute node
export BML_NONMPI_PRECOMMAND=${BML_NONMPI_PRECOMMAND:=${JSRUN}}
export BML_NONMPI_PRECOMMAND_ARGS=${BML_NONMPI_PRECOMMAND_ARGS:="-n1;-a1;-g1;-c7"}

./build.sh install
