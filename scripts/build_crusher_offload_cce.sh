#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)

export CC=${CC:=cc}
export FC=${FC:=ftn}
export CXX=${CXX:=CC}
export BLA_VENDOR=${BLA_VENDOR:=OpenBLAS}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_OMP_OFFLOAD=${BML_OMP_OFFLOAD:=yes}
export BML_OFFLOAD_ARCH=${BML_OFFLOAD_ARCH:=AMD}
export BUILD_DIR=${BUILD_DIR:="${MY_PATH}/build"}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}

./build.sh configure

pushd ${BUILD_DIR}
make -j16
make install
popd
