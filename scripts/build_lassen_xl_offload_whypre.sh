#!/bin/bash

# Make sure all the paths are correct

source scripts/setenv_lassen_offload.sh

rm -r build
rm -r install

MY_PATH=$(pwd)

HYPRE_INSTALL_PATH="/usr/WS1/osei/soft/CoPA/lassen/gpu/fork/bml-hypre/hypre/src/hypre"

export CC=${CC:=xlc-gpu}
export FC=${FC:=xlf2003-gpu}
export CXX=${CXX:=xlc++-gpu}
export BLAS_VENDOR=${BLAS_VENDOR:=Auto}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_OMP_OFFLOAD=${BML_OMP_OFFLOAD:=yes}
export BML_OFFLOAD_ARCH=${BML_OFFLOAD_ARCH:=NVIDIA}
export BML_CUSPARSE=${BML_CUSPARSE:=no}
export BML_HYPRE=${BML_HYPRE:=yes}
export HYPRE_ROOT=${HYPRE_INSTALL_PATH}
export BML_COMPLEX=${BML_COMPLEX:=no}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:=""}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:=""}
#export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-lm -L/usr/tce/packages/xl/xl-2021.03.11/xlC/16.1.1/lib -libmc++"}
export BLAS_LIBRARIES=${BLAS_LIBRARIES:="-L${ESSLLIBDIR64} -lesslsmp"}
export LAPACK_LIBRARIES=${LAPACK_LIBRARIES:="-L${LAPACK_DIR} -llapack"}

export CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME}

./build.sh configure

pushd build
make -j
make install
popd
