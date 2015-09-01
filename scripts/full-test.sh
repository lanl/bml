#!/bin/sh

BUILD_DIR=${PWD}/build
INSTALL_DIR=${PWD}/install

mkdir -p "${BUILD_DIR}" || exit
mkdir -p "${INSTALL_DIR}" || exit

cd "${BUILD_DIR}"
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_Fortran_COMPILER=${FC:=gfortran} \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DBML_TESTING=yes

make || exit
make doc || exit
make test || exit
make install || exit
