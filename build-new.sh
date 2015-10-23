#!/bin/bash

TOP_DIR="${PWD}"
BUILD_DIR="${TOP_DIR}/build-new"
INSTALL_DIR="${INSTALL_DIR:=${TOP_DIR}/install-new}"
LOG_FILE="${TOP_DIR}/build-new.log"

create() {
    mkdir -v -p "${BUILD_DIR}" || exit
    mkdir -v -p "${INSTALL_DIR}" || exit
}

configure() {
    cd "${BUILD_DIR}"
    ${CMAKE:=cmake} .. \
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Debug} \
          -DCMAKE_C_COMPILER=${CC:=gcc} \
          $([[ -n ${CMAKE_C_FLAGS} ]] \
          && echo "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}") \
          -DCMAKE_CXX_COMPILER=${CXX:=g++} \
          $([[ -n ${CMAKE_CXX_FLAGS} ]] \
          && echo "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}") \
          -DCMAKE_Fortran_COMPILER=${FC:=gfortran} \
          $([[ -n ${CMAKE_Fortran_FLAGS} ]] \
          && echo "-DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}") \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
          -DBML_OPENMP=${BML_OPENMP:=yes} \
          -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:=no} \
          -DBML_TESTING=${BML_TESTING:=yes} \
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

commands=("create" "configure" "compile" "install" "testing" "docs")

if [[ $# -gt 0 ]]; then
    if [[ "$1" = "-h" || "$1" = "--help" ]]; then
        cat <<EOF
Usage:

This script can be used to build and test the bml library. When run
without arguments, it will create the subdirectories 'build' and
'install', configure and build the library in 'build', run all tests,
and install it in 'install'. If called with a command, each step can
be executed separately, including all necessary previous steps. Known
commands are:

create
configure
compile
install
testing
docs
EOF
        exit 0
    fi

    is_legal=0
    for c in ${commands[@]}; do
        if [[ "$1" = "${c}" ]]; then
            is_legal=1
            break
        fi
    done
    if [[ ${is_legal} -ne 1 ]]; then
        echo "unknown command $1"
        exit 1
    fi

    if [[ "$1" = "create" ]]; then
        create
    elif [[ "$1" = "configure" ]]; then
        create
        configure
    elif [[ "$1" = "docs" ]]; then
        create
        configure
        docs
    elif [[ "$1" = "compile" ]]; then
        create
        configure
        compile
    elif [[ "$1" = "install" ]]; then
        create
        configure
        install
    elif [[ "$1" = "testing" ]]; then
        create
        configure
        compile
        testing
    else
        echo "unknown command $1"
        exit 1
    fi
else
    create
    configure
    #docs
    compile
    #install
    #testing
fi

echo "The output was written to ${LOG_FILE}"
