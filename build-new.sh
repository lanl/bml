#!/bin/sh

BUILD_DIR=${PWD}/build-new
INSTALL_DIR=${PWD}/install-new

mkdir -p "${BUILD_DIR}" || exit
mkdir -p "${INSTALL_DIR}" || exit

cd "${BUILD_DIR}"
cmake .. \
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Debug} \
  -DCMAKE_C_COMPILER=${CC:=gcc} \
  -DCMAKE_CXX_COMPILER=${CXX:=g++} \
  -DCMAKE_Fortran_COMPILER=${FC:=gfortran} \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DBML_TESTING=yes \
  -DBML_NEW=yes

make || exit
make docs || exit
ctest --output-on-failure
make install
