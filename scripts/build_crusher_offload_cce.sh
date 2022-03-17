#!/bin/bash

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)

export CC=${CC:=cc}
export FC=${FC:=ftn}
export CXX=${CXX:=CC}
export BML_OPENMP=${BML_OPENMP:=yes}
export BUILD_DIR=${BUILD_DIR:="${MY_PATH}/build"}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-ef -DCRAY_SDK"}
export CMAKE_C_FLAGS=${CMAKE_C_FLAGS:="-Ofast -DUSE_OMP_OFFLOAD -DCRAY_SDK"}
export BLAS_LIBRARIES=${BLAS_LIBRARIES:="-L${LIBSCI_BASE_DIR}/cray/9.0/x86_64/lib -lsci_cray"}

./build.sh configure

pushd ${BUILD_DIR}
make -j16
make install
popd
