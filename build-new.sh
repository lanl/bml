#!/bin/bash

TOP_DIR="${PWD}"
BUILD_DIR="${TOP_DIR}/build-new"
INSTALL_DIR="${INSTALL_DIR:=${TOP_DIR}/install-new}"
LOG_FILE="${TOP_DIR}/build-new.log"

create_directories() {
    mkdir -v -p "${BUILD_DIR}" || exit
    mkdir -v -p "${INSTALL_DIR}" || exit
}

configure() {
    cd "${BUILD_DIR}"
    cmake .. \
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Debug} \
          -DCMAKE_C_COMPILER=${CC:=gcc} \
          -DCMAKE_CXX_COMPILER=${CXX:=g++} \
          -DCMAKE_Fortran_COMPILER=${FC:=gfortran} \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
          -DBML_OPENMP=${BML_OPENMP:=yes} \
          -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:=no} \
          -DBML_TESTING=${BML_TESTIND:=yes} \
          -DBLAS_VENDOR=${BLAS_VENDOR:=} \
          -DBML_NEW=yes | tee -a "${LOG_FILE}" || exit
    cd "${TOP_DIR}"
}

compile() {
    make -C "${BUILD_DIR}" VERBOSE=1 2>&1 | tee -a "${LOG_FILE}" || exit
}

docs() {
    make -C "${BUILD_DIR}" docs 2>&1 | tee -a "${LOG_FILE}" || exit
    make -C "${BUILD_DIR}/doc/latex" 2>&1 | tee -a "${LOG_FILE}" || exit
    if test -f "${BUILD_DIR}/doc/latex/refman.pdf"; then
        cp -v "${BUILD_DIR}/doc/latex/refman.pdf" "${TOP_DIR}/bml-manual.pdf"
    fi
}

install() {
    make -C "${BUILD_DIR}" install 2>&1 | tee -a "${LOG_FILE}"
}

testing() {
    cd "${BUILD_DIR}"
    ctest --output-on-failure 2>&1 | tee -a "${LOG_FILE}"
    cd "${TOP_DIR}"
}

if [[ $# -gt 0 ]]; then
    if [[ "$1" = "create_directories" ]]; then
        create_directories
    elif [[ "$1" = "configure" ]]; then
        create_directories
        configure
    elif [[ "$1" = "docs" ]]; then
        create_directories
        configure
        docs
    elif [[ "$1" = "compile" ]]; then
        create_directories
        configure
        compile
    elif [[ "$1" = "install" ]]; then
        create_directories
        configure
        install
    elif [[ "$1" = "testing" ]]; then
        create_directories
        configure
        compile
        testing
    else
        echo "unknown command $1"
        exit 1
    fi
else
    create_directories
    configure
    docs
    compile
    install
    testing
fi

echo "The output was written to ${LOG_FILE}"
