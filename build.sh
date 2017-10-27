#!/bin/bash

TOP_DIR="$(readlink --canonicalize-existing $(dirname "$0"))"
: ${BUILD_DIR:=${TOP_DIR}/build}
: ${INSTALL_DIR:=${TOP_DIR}/install}
LOG_FILE="${TOP_DIR}/build.log"
: ${VERBOSE_MAKEFILE:=no}
: ${PARALLEL_TEST_JOBS:=1}

help() {
    cat <<EOF
Usage:

This script can be used to build and test the bml library.  The script has to
be given a command. Known commands are:

cleanup         - Clean up and remove build and install directories
create          - Create the build and install directories
configure       - Configure the build system
compile         - Compile the sources
install         - Install the compiled sources
testing         - Run the test suite
docs            - Generate the API documentation
indent          - Indent the sources
check_indent    - Check the indentation of the sources
tags            - Create tags file for vim and emacs
dist            - Generate a tar file (this only works with git)

The following environment variables can be set to influence the configuration
step and the build:

EOF
    set_defaults
    echo "CMAKE_BUILD_TYPE     {Release,Debug}             (default is ${CMAKE_BUILD_TYPE})"
    echo "CC                   Path to C compiler          (default is ${CC})"
    echo "CXX                  Path to C++ compiler        (default is ${CXX})"
    echo "FC                   Path to Fortran compiler    (default is ${FC})"
    echo "BML_OPENMP           {yes,no}                    (default is ${BML_OPENMP})"
    echo "BML_MPI              {yes,no}                    (default is ${BML_MPI})"
    echo "BML_TESTING          {yes,no}                    (default is ${BML_TESTING})"
    echo "BUILD_DIR            Path to build dir           (default is ${BUILD_DIR})"
    echo "BLAS_VENDOR          {,Intel,MKL,ACML,GNU}       (default is '${BLAS_VENDOR}')"
    echo "BML_INTERNAL_BLAS    {yes,no}                    (default is ${BML_INTERNAL_BLAS})"
    echo "INSTALL_DIR          Path to install dir         (default is ${INSTALL_DIR})"
    echo "EXTRA_CFLAGS         Extra C flags               (default is '${EXTRA_CFLAGS}')"
    echo "EXTRA_FFLAGS         Extra fortran flags         (default is '${EXTRA_FFLAGS}')"
    echo "PARALLEL_TEST_JOBS   The number of test jobs     (default is ${PARALLEL_TEST_JOBS})"
    echo "CFLAGS               Set C compiler flags        (default is '${CFLAGS}')"
    echo "CXXFLAGS             Set C++ compiler flags      (default is '${CXXFLAGS}')"
    echo "FFLAGS               Set Fortran compiler flags  (default is '${CFLAGS}')"
    echo "EXTRA_LINK_FLAGS     Add extra link flags        (default is '${EXTRA_LINK_FLAGS}')"
}

set_defaults() {
    : ${CMAKE_BUILD_TYPE:=Release}
    : ${CC:=gcc}
    : ${CXX:=g++}
    : ${FC:=gfortran}
    : ${BML_OPENMP:=yes}
    : ${BML_MPI:=no}
    : ${BLAS_VENDOR:=}
    : ${BML_INTERNAL_BLAS:=no}
    : ${EXTRA_CFLAGS:=""}
    : ${EXTRA_FFLAGS:=""}
    : ${BML_TESTING:=yes}
    : ${FORTRAN_FLAGS:=""}
    : ${FFLAGS:=""}
    : ${CFLAGS:=""}
    : ${CXXFLAGS:=""}
    : ${EXTRA_LINK_FLAGS:=""}
}

die() {
    echo "fatal error"
    if [[ -f "${BUILD_DIR}/CMakeFiles/CMakeOutput.log" ]]; then
        echo "appending CMake output"
        echo "*********** CMake Output ***********" >> ${LOG_FILE}
        cat "${BUILD_DIR}/CMakeFiles/CMakeOutput.log" >> ${LOG_FILE}
    fi
    if [[ -f "${BUILD_DIR}/CMakeFiles/CMakeError.log" ]]; then
        echo "appending CMake error"
        echo "*********** CMake Error ***********" >> ${LOG_FILE}
        cat "${BUILD_DIR}/CMakeFiles/CMakeError.log" >> ${LOG_FILE}
    fi
    echo "the output from this build was written to ${LOG_FILE}"
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

cleanup() {
    rm -vrf "${BUILD_DIR}" || die
    rm -vrf "${INSTALL_DIR}" || die
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
        -DBML_INTERNAL_BLAS="${BML_INTERNAL_BLAS}" \
        -DEXTRA_CFLAGS="${EXTRA_CFLAGS}" \
        -DEXTRA_FFLAGS="${EXTRA_FFLAGS}" \
        -DCMAKE_Fortran_FLAGS="${FFLAGS}" \
        -DCMAKE_C_FLAGS="${CFLAGS}" \
        -DCMAKE_CXX_FLAGS="${CXXLAGS}" \
        -DCMAKE_VERBOSE_MAKEFILE=${VERBOSE_MAKEFILE} \
        -DBML_LINK_FLAGS=${EXTRA_LINK_FLAGS} \
        | tee -a "${LOG_FILE}"
    check_pipe_error
    cd "${TOP_DIR}"
}

compile() {
    make -C "${BUILD_DIR}" | tee -a "${LOG_FILE}"
    check_pipe_error
}

docs() {
    make -C "${BUILD_DIR}" docs 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
    #make -C "${BUILD_DIR}/doc/latex" 2>&1 | tee -a "${LOG_FILE}"
    #check_pipe_error
    #if test -f "${BUILD_DIR}/doc/latex/refman.pdf"; then
    #cp -v "${BUILD_DIR}/doc/latex/refman.pdf" "${TOP_DIR}/bml-manual.pdf"
    #fi
}

install() {
    make -C "${BUILD_DIR}" install 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

testing() {
    cd "${BUILD_DIR}"
    ctest --output-on-failure --parallel ${PARALLEL_TEST_JOBS} 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
    cd "${TOP_DIR}"
}

indent() {
    cd "${BUILD_DIR}"
    "${TOP_DIR}/indent.sh" 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

check_indent() {
    cd "${BUILD_DIR}"
    "${TOP_DIR}/indent.sh" 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
    cd "${TOP_DIR}"
    git diff 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
    LINES=$(git diff | wc -l)
    if test ${LINES} -gt 0; then
        echo "sources were not formatted correctly"
        die
    fi
}

tags() {
    "${TOP_DIR}/update_tags.sh" 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

dist() {
    make -C "${BUILD_DIR}" dist 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

echo "Writing output to ${LOG_FILE}"

if [[ $# -gt 0 ]]; then
    if [[ "$1" = "-h" || "$1" = "--help" ]]; then
        help
        shift
    fi

    while [[ $# -gt 0 ]]; do
        echo "Running command $1"
        case "$1" in
            "cleanup")
                cleanup
                ;;
            "create")
                create
                ;;
            "configure")
                create
                configure
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
            "docs")
                create
                configure
                docs
                ;;
            "indent")
                indent
                ;;
            "check_indent")
                create
                check_indent
                ;;
            "tags")
                tags
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
        shift
    done
else
    echo "missing action"
    help
fi

echo "The output was written to ${LOG_FILE}"
