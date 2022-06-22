#!/bin/bash
module load cmake
module load cuda
module load gcc/11.2.0
module load openblas

rm -r build
rm -r install

MY_PATH=$(pwd)

#get jsrun with full path
JSRUN=$(which jsrun)
echo ${JSRUN}

export CC=${CC:=gcc}
export FC=${FC:=gfortran}
export CXX=${CXX:=g++}
export BLAS_VENDOR=${BLAS_VENDOR:=OpenBLAS}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_OMP_OFFLOAD=${BML_OMP_OFFLOAD:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export BML_COMPLEX=${BML_COMPLEX:=no}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-latomic"}
export BML_CUSPARSE=${BML_CUSPARSE:=yes}
export BML_COMPLEX=${BML_COMPLEX:=no}

#use jsrun to run tests on a compute node
export BML_NONMPI_PRECOMMAND=${BML_NONMPI_PRECOMMAND:=${JSRUN}}
export BML_NONMPI_PRECOMMAND_ARGS=${BML_NONMPI_PRECOMMAND_ARGS:="-n1;-a1;-g1;-c7"}

./build.sh install
