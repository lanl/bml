#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)

export CC=${CC:=cc}
export FC=${FC:=ftn}
export CXX=${CXX:=CC}

export BLAS_VENDOR=${BLAS_VENDOR:=OpenBLAS}

export BML_OPENMP=${BML_OPENMP:=yes}

export BML_MAGMA=${BML_MAGMA:=yes}
export MAGMA_ROOT=${OLCF_MAGMA_ROOT}
export BML_ROCSOLVER=${BML_ROCSOLVER:=no}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export INSTALL_DIR="/autofs/nccs-svm1_proj/csc304/bml/crusher/cce"
export EXTRA_FFLAGS=${EXTRA_FCFLAGS:="-hsystem_alloc"}

export BML_NONMPI_PRECOMMAND=${BML_NONMPI_PRECOMMAND:="srun"}
export BML_NONMPI_PRECOMMAND_ARGS=${BML_NONMPI_PRECOMMAND_ARGS:="-n1;-c4;--gpus=1"}

./build.sh configure

pushd build
make -j8
make install
popd
