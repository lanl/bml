#!/bin/bash

TOP_DIR="${PWD}"
BUILD_DIR="${BUILD_DIR:=${TOP_DIR}/build}"
INSTALL_DIR="${INSTALL_DIR:=${TOP_DIR}/install}"
LOG_FILE="${TOP_DIR}/build.log"
VERBOSE_MAKEFILE="${VERBOSE_MAKEFILE:=no}"

help() {
    cat <<EOF
Usage:

This script can be used to build and test the bml library.  The script has to
be given a command. Known commands are:

create     - Create the build and install directories ('build' and 'install')
configure  - Configure the build system
compile    - Compile the sources
install    - Install the compiled sources
testing    - Run the test suite
docs       - Generate the API documentation
dist       - Generate a tar file (this only works with git)
The following environment variables can be set to influence the configuration
step and the build:

EOF
    set_defaults
    echo "CMAKE_BUILD_TYPE   {Release,Debug}          (default is ${CMAKE_BUILD_TYPE})"
    echo "CC                 Path to C compiler       (default is ${CC})"
    echo "CXX                Path to C++ compiler     (default is ${CXX})"
    echo "FC                 Path to Fortran compiler (default is ${FC})"
    echo "BLAS_VENDOR        {,Intel,MKL,ACML}        (default is '${BLAS_VENDOR}')"
    echo "BML_OPENMP         {yes,no}                 (default is ${BML_OPENMP})"
    echo "BML_MPI            {yes,no}                 (default is ${BML_MPI})"
    echo "BML_TESTING        {yes,no}                 (default is ${BML_TESTING})"
    echo "BUILD_DIR          Path to build dir        (default is ${BUILD_DIR})"
}

set_defaults() {
    CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Debug}
    CC="${CC:=gcc}"
    CXX="${CXX:=g++}"
    FC="${FC:=gfortran}"
    BML_OPENMP=${BML_OPENMP:=yes}
    BML_MPI=${BML_MPI:=no}
    BLAS_VENDOR="${BLAS_VENDOR:=}"
    BML_TESTING=${BML_TESTING:=yes}
}

die() {
  echo "fatal error"
  if [[ $# -gt 1 ]]; then
    exit $1
  else
    exit 1
  fi
}

check_pipe_error() {
  if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    die ${PIPESTATUS[0]}
  fi
}

create() {
    mkdir -v -p "${BUILD_DIR}" || die
    mkdir -v -p "${INSTALL_DIR}" || die
}

configure() {
    set_defaults
    cd "${BUILD_DIR}"
    if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
      rm -v "${BUILD_DIR}/CMakeCache.txt" || die
    fi
    ${CMAKE:=cmake} .. \
          -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
          -DCMAKE_C_COMPILER="${CC}" \
          -DCMAKE_CXX_COMPILER="${CXX}" \
          -DCMAKE_Fortran_COMPILER="${FC}" \
          $([[ -n ${CMAKE_C_FLAGS} ]] && echo "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}") \
          $([[ -n ${CMAKE_CXX_FLAGS} ]] && echo "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}") \
          $([[ -n ${CMAKE_Fortran_FLAGS} ]] && echo "-DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}") \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
          -DBML_OPENMP="${BML_OPENMP}" \
          -DBML_MPI="${BML_MPI}" \
          -DBUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:=no}" \
          -DBML_TESTING="${BML_TESTING:=yes}" \
          -DBLAS_VENDOR="${BLAS_VENDOR}" \
        | tee -a "${LOG_FILE}"
    check_pipe_error
    cd "${TOP_DIR}"
}

compile() {
    make -C "${BUILD_DIR}" \
      $([[ "${VERBOSE_MAKEFILE}" = "yes" ]] && echo "VERBOSE=1") 2>&1 \
      | tee -a "${LOG_FILE}"
    check_pipe_error
}

docs() {
    make -C "${BUILD_DIR}" docs 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
    make -C "${BUILD_DIR}/doc/latex" 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
    if test -f "${BUILD_DIR}/doc/latex/refman.pdf"; then
        cp -v "${BUILD_DIR}/doc/latex/refman.pdf" "${TOP_DIR}/bml-manual.pdf"
    fi
}

install() {
    make -C "${BUILD_DIR}" install 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

testing() {
    cd "${BUILD_DIR}"
    ctest --output-on-failure 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
    cd "${TOP_DIR}"
}

dist() {
    make -C "${BUILD_DIR}" dist 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

if [[ $# -gt 0 ]]; then
    if [[ "$1" = "-h" || "$1" = "--help" ]]; then
        help
        exit 0
    fi

    case "$1" in
        "create")
            create
            ;;
        "configure")
            create
            configure
            ;;
        "docs")
            create
            configure
            docs
            ;;
        "compile")
            create
            configure
            compile
            ;;
        "install")
            create
            configure
            install
            ;;
        "testing")
            create
            configure
            compile
            testing
            ;;
        "dist")
            create
            configure
            dist
            ;;
        *)
            echo "unknown command $1"
            exit 1
            ;;
    esac
else
    echo "missing action"
    help
fi

echo "The output was written to ${LOG_FILE}"
