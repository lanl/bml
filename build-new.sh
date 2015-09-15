#!/bin/sh

TOP_DIR="${PWD}"
BUILD_DIR="${TOP_DIR}"/build-new
INSTALL_DIR="${TOP_DIR}"/install-new
LOG_FILE="${TOP_DIR}"/build-new.log

mkdir -v -p "${BUILD_DIR}" || exit
mkdir -v -p "${INSTALL_DIR}" || exit

cd "${BUILD_DIR}"
cmake .. \
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Debug} \
  -DCMAKE_C_COMPILER=${CC:=gcc} \
  -DCMAKE_CXX_COMPILER=${CXX:=g++} \
  -DCMAKE_Fortran_COMPILER=${FC:=gfortran} \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DBML_TESTING=yes \
  -DBML_NEW=yes
cd "${TOP_DIR}"

make -C "${BUILD_DIR}" VERBOSE=1 2>&1 | tee --append "${LOG_FILE}" || exit
make -C "${BUILD_DIR}" docs 2>&1 | tee --append "${LOG_FILE}" || exit
cd "${BUILD_DIR}"
ctest --output-on-failure 2>&1 | tee --append "${LOG_FILE}"
cd "${TOP_DIR}"
make -C "${BUILD_DIR}" install 2>&1 | tee --append "${LOG_FILE}"

echo "The output was written to ${LOG_FILE}"
