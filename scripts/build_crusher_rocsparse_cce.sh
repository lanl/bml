#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)

export CC=${CC:=cc}
export FC=${FC:=ftn}
export CXX=${CXX:=CC}
export BLA_VENDOR=${BLA_VENDOR:=OpenBLAS}
export BML_OMP_OFFLOAD=${BML_OMP_OFFLOAD:=yes}
export BML_OFFLOAD_ARCH=${BML_OFFLOAD_ARCH:=AMD}
export BML_ROCSPARSE=${BML_ROCSPARSE:=yes}
export BML_COMPLEX=${BML_COMPLEX:=no}
export BML_OPENMP=${BML_OPENMP:=yes}
export BUILD_DIR=${BUILD_DIR:="${MY_PATH}/build"}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
#export EXTRA_CFLAGS=${EXTRA_CFLAGS:=-craype-verbose}
./build.sh configure

pushd ${BUILD_DIR}
make -j16
make install
popd
