#!/bin/bash

set -u

TOP_DIR="$(dirname "$0")"
TOP_DIR="$(readlink --canonicalize-existing ${TOP_DIR} 2> /dev/null)"
if (( $? != 0 )); then
    # Fall back to bash function `pwd`. Note that this fallback
    # depends on using bash.
    TOP_DIR=$(pwd -P $TOP_DIR)
fi

: ${CMAKE:=cmake}
: ${BUILD_DIR:=${TOP_DIR}/build}
: ${INSTALL_DIR:=${TOP_DIR}/install}
: ${PARALLEL_TEST_JOBS:=1}
: ${TESTING_EXTRA_ARGS:=}
: ${VERBOSE_MAKEFILE:=no}
LOG_FILE="${TOP_DIR}/build.log"

help() {
    cat <<EOF
Usage:

build.sh [--debug] COMMAND

This script can be used to build and test the bml library. The script
has to be given a COMMAND. Known commands are:

cleanup         - Clean up and remove build and install directories
create          - Create the build and install directories
configure       - Configure the build system
compile         - Compile the sources
install         - Install the compiled sources
testing         - Run the test suite
docs            - Generate the API documentation
indent          - Check the indentation of the sources
check_indent    - Check the indentation of the sources
tags            - Create tags file for vim and emacs
dist            - Generate a tar file (this only works with git)

The following environment variables can be set to influence the
configuration step and the build:

EOF
    set_defaults
    echo "CMAKE_BUILD_TYPE           {Release,Debug}             (default is ${CMAKE_BUILD_TYPE})"
    echo "BUILD_SHARED_LIBS          Build a shared library      (default is ${BUILD_SHARED_LIBS})"
    echo "CC                         Path to C compiler          (default is ${CC})"
    echo "CXX                        Path to C++ compiler        (default is ${CXX})"
    echo "FC                         Path to Fortran compiler    (default is ${FC})"
    echo "BML_OPENMP                 {yes,no}                    (default is ${BML_OPENMP})"
    echo "MKL_GPU                    {yes,no}                    (default is ${MKL_GPU})"
    echo "BML_MPI                    {yes,no}                    (default is ${BML_MPI})"
    echo "BML_MPIEXEC_EXECUTABLE     Command to prepend MPI tests"
    echo "                           (default is '${BML_MPIEXEC_EXECUTABLE}')"
    echo "BML_MPIEXEC_NUMPROCS_FLAG  Flags to specify number of MPI tasks for MPI tests"
    echo "                           (default is ${BML_MPIEXEC_NUMPROCS_FLAG})"
    echo "BML_MPIEXEC_NUMPROCS       Number of MPI tasks for MPI tests"
    echo "                           (default is ${BML_MPIEXEC_NUMPROCS})"
    echo "BML_MPIEXEC_PREFLAGS       Extra flags for MPI tests   (default is '${BML_MPIEXEC_PREFLAGS}')"
    echo "BML_COMPLEX                {yes,no}                    (default is ${BML_COMPLEX})"
    echo "BML_TESTING                {yes,no}                    (default is ${BML_TESTING})"
    echo "BML_VALGRIND               {yes,no}                    (default is ${BML_VALGRIND})"
    echo "BML_COVERAGE               {yes,no}                    (default is ${BML_COVERAGE})"
    echo "BML_NONMPI_PRECOMMAND      Command to prepend to tests (default is ${BML_NONMPI_PRECOMMAND})"
    echo "BML_NONMPI_PRECOMMAND_ARGS Arguments for prepend command"
    echo "                           (default is ${BML_NONMPI_PRECOMMAND_ARGS})"
    echo "BUILD_DIR                  Path to build dir           (default is ${BUILD_DIR})"
    echo "BLA_VENDOR or BLAS_VENDOR  Any vendor defined in"
    echo "                           FindBLAS.cmake https://cmake.org/cmake/help/latest/module/FindBLAS.html"
    echo "                           (default is '${BLAS_VENDOR}')"
    echo "BML_INTERNAL_BLAS          {yes,no}                    (default is ${BML_INTERNAL_BLAS})"
    echo "PARALLEL_TEST_JOBS         The number of test jobs     (default is ${PARALLEL_TEST_JOBS})"
    echo "TESTING_EXTRA_ARGS         Arguments to ctest, e.g. '-R C-.*-double_real'"
    echo "INSTALL_DIR                Path to install dir         (default is ${INSTALL_DIR})"
    echo "CMAKE_C_FLAGS              Set C compiler flags        (default is '${CMAKE_C_FLAGS}')"
    echo "CMAKE_CXX_FLAGS            Set C++ compiler flags      (default is '${CMAKE_CXX_FLAGS}')"
    echo "CMAKE_Fortran_FLAGS        Set Fortran compiler flags  (default is '${CMAKE_Fortran_FLAGS}')"
    echo "BLAS_LIBRARIES             Blas libraries              (default is '${BLAS_LIBRARIES}')"
    echo "LAPACK_LIBRARIES           Lapack libraries            (default is '${LAPACK_LIBRARIES}')"
    echo "EXTRA_CFLAGS               Extra C flags               (default is '${EXTRA_CFLAGS}')"
    echo "EXTRA_FFLAGS               Extra fortran flags         (default is '${EXTRA_FFLAGS}')"
    echo "EXTRA_LINK_FLAGS           Add extra link flags        (default is '${EXTRA_LINK_FLAGS}')"
    echo "BML_OMP_OFFLOAD            {yes,no}                    (default is ${BML_OMP_OFFLOAD})"
    echo "BML_OFFLOAD_ARCH           {NVIDIA, AMD}               (default is ${BML_OFFLOAD_ARCH})"
    echo "GPU_ARCH                   GPU architecture            (default is '${GPU_ARCH}')"
    echo "BML_CUDA                   Build with CUDA             (default is ${BML_CUDA})"
    echo "BML_MAGMA                  Build with MAGMA            (default is ${BML_MAGMA})"
    echo "BML_CUSOLVER               Build with cuSOLVER         (default is ${BML_CUSOLVER})"
    echo "BML_CUSPARSE               Build with cuSPARSE         (default is ${BML_CUSPARSE})"
    echo "BML_ROCSOLVER              Build with rocSOLVER        (default is ${BML_ROCSOLVER})"
    echo "BML_ROCSPARSE              Build with rocSPARSE        (default is ${BML_ROCSPARSE})"
    echo "BML_XSMM                   Build with XSMM             (default is ${BML_XSMM})"
    echo "BML_SCALAPACK              Build with SCALAPACK        (default is ${BML_SCALAPACK})"
    echo "SCALAPACK_LIBRARIES        ScaLapack libraries         (default is '${SCALAPACK_LIBRARIES}')"
    echo "BML_ELLBLOCK_MEMPOOL       Use ellblock memory pool    (default is ${BML_ELLBLOCK_MEMPOOL}"
    echo "CUDA_TOOLKIT_ROOT_DIR      Path to CUDA dir            (default is '${CUDA_TOOLKIT_ROOT_DIR}')"
    echo "INTEL_OPT                  {yes, no}                   (default is ${INTEL_OPT})"
    echo "CMAKE_ARGS                 pass-through CMake flags    (default is '${CMAKE_ARGS}')"
}

set_defaults() {
    : ${CMAKE_BUILD_TYPE:=Release}
    : ${BUILD_SHARED_LIBS:=no}
    : ${CC:=gcc}
    : ${CXX:=g++}
    : ${FC:=gfortran}
    : ${BML_OPENMP:=yes}
    : ${MKL_GPU:=no}
    : ${BML_MPI:=no}
    : ${BML_MPIEXEC_EXECUTABLE:=}
    : ${BML_MPIEXEC_NUMPROCS_FLAG:=-n}
    : ${BML_MPIEXEC_NUMPROCS:=4}
    : ${BML_MPIEXEC_PREFLAGS:=}
    : ${BML_COMPLEX:=yes}
    : ${BLAS_VENDOR:=}
    : ${BLA_VENDOR:=}
    : ${BML_INTERNAL_BLAS:=no}
    : ${EXTRA_CFLAGS:=}
    : ${EXTRA_FFLAGS:=}
    : ${CMAKE_C_FLAGS:=}
    : ${CMAKE_CXX_FLAGS:=}
    : ${CMAKE_Fortran_FLAGS:=}
    : ${BLAS_LIBRARIES:=}
    : ${LAPACK_LIBRARIES:=}
    : ${SCALAPACK_LIBRARIES:=}
    : ${BML_TESTING:=yes}
    : ${BML_VALGRIND:=no}
    : ${BML_COVERAGE:=no}
    : ${BML_NONMPI_PRECOMMAND:=}
    : ${BML_NONMPI_PRECOMMAND_ARGS:=}
    : ${FORTRAN_FLAGS:=}
    : ${EXTRA_LINK_FLAGS:=}
    : ${BML_OMP_OFFLOAD:=no}
    : ${BML_OFFLOAD_ARCH:=NVIDIA}
    : ${GPU_ARCH:=}
    : ${BML_CUDA:=no}
    : ${BML_MAGMA:=no}
    : ${BML_CUSOLVER:=no}
    : ${BML_CUSPARSE:=no}
    : ${BML_ROCSOLVER:=no}
    : ${BML_ROCSPARSE:=no}
    : ${BML_XSMM:=no}
    : ${BML_SCALAPACK:=no}
    : ${BML_ELLBLOCK_MEMPOOL:=no}
    : ${CUDA_TOOLKIT_ROOT_DIR:=}
    : ${INTEL_OPT:=no}
    : ${CMAKE_ARGS:=}
}

die() {
    if (( $# > 0 )); then
        echo "*********************************************"
        echo "$@"
        echo "*********************************************"
    fi
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

is_enabled() {
    if [[ $1 == yes || $1 == on || $1 == 1 ]]; then
        return 0
    else
        return 1
    fi
}

sanity_check() {
    if (is_enabled ${BML_CUSPARSE} || is_enabled ${BML_ROCSPARSE}) && ! is_enabled ${BML_OMP_OFFLOAD}; then
        die "In order to enable BML_CUSPARSE or BML_ROCSPARSE, BML_OMP_OFFLOAD needs to be enabled as well"
    fi
}

configure() {
    cd "${BUILD_DIR}"
    if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        rm -v "${BUILD_DIR}/CMakeCache.txt" || die
    fi
    ${CMAKE} .. \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DCMAKE_Fortran_COMPILER="${FC}" \
        ${CMAKE_C_FLAGS:+-DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"} \
        ${CMAKE_CXX_FLAGS:+-DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"} \
        ${CMAKE_Fortran_FLAGS:+-DCMAKE_Fortran_FLAGS="${CMAKE_Fortran_FLAGS}"} \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DBLAS_LIBRARIES="${BLAS_LIBRARIES}" \
        -DLAPACK_LIBRARIES="${LAPACK_LIBRARIES}" \
        -DSCALAPACK_LIBRARIES="${SCALAPACK_LIBRARIES}" \
        -DBML_OPENMP="${BML_OPENMP}" \
        -DMKL_GPU="${MKL_GPU}" \
        -DBML_MPI="${BML_MPI}" \
        -DBML_MPIEXEC_EXECUTABLE="${BML_MPIEXEC_EXECUTABLE}" \
        -DBML_MPIEXEC_NUMPROCS_FLAG="${BML_MPIEXEC_NUMPROCS_FLAG}" \
        -DBML_MPIEXEC_NUMPROCS="${BML_MPIEXEC_NUMPROCS}" \
        -DBML_MPIEXEC_PREFLAGS="${BML_MPIEXEC_PREFLAGS}" \
        -DBML_COMPLEX="${BML_COMPLEX}" \
        -DBUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
        -DBML_TESTING="${BML_TESTING:=yes}" \
        -DBML_VALGRIND="${BML_VALGRIND:=no}" \
        -DBML_COVERAGE="${BML_COVERAGE:=no}" \
        -DBML_NONMPI_PRECOMMAND="${BML_NONMPI_PRECOMMAND}" \
        -DBML_NONMPI_PRECOMMAND_ARGS="${BML_NONMPI_PRECOMMAND_ARGS}" \
        -DBLAS_VENDOR="${BLAS_VENDOR}" \
        -DBML_INTERNAL_BLAS="${BML_INTERNAL_BLAS}" \
        ${EXTRA_CFLAGS:+-DEXTRA_CFLAGS="${EXTRA_CFLAGS}"} \
        ${EXTRA_FFLAGS:+-DEXTRA_FFLAGS="${EXTRA_FFLAGS}"} \
        ${EXTRA_LINK_FLAGS:+-DBML_LINK_FLAGS="${EXTRA_LINK_FLAGS}"} \
        -DCMAKE_VERBOSE_MAKEFILE=${VERBOSE_MAKEFILE} \
        -DBML_OMP_OFFLOAD="${BML_OMP_OFFLOAD}" \
        -DBML_OFFLOAD_ARCH="${BML_OFFLOAD_ARCH}" \
        -DGPU_ARCH="${GPU_ARCH}" \
        -DBML_CUDA="${BML_CUDA}" \
        -DBML_MAGMA="${BML_MAGMA}" \
        -DBML_CUSOLVER="${BML_CUSOLVER}" \
        -DBML_CUSPARSE="${BML_CUSPARSE}" \
        -DBML_ROCSOLVER="${BML_ROCSOLVER}" \
        -DBML_ROCSPARSE="${BML_ROCSPARSE}" \
        -DBML_XSMM="${BML_XSMM}" \
        -DBML_SCALAPACK="${BML_SCALAPACK}" \
        -DBML_ELLBLOCK_MEMPOOL="${BML_ELLBLOCK_MEMPOOL}" \
        -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}" \
        -DINTEL_OPT="${INTEL_OPT:=no}" \
        ${CMAKE_ARGS} \
        | tee -a "${LOG_FILE}"
    check_pipe_error
    cd "${TOP_DIR}"
}

compile() {
    ${CMAKE} --build "${BUILD_DIR}" | tee -a "${LOG_FILE}"
    check_pipe_error
}

docs() {
    ${CMAKE} --build "${BUILD_DIR}" --target docs 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
    #make -C "${BUILD_DIR}/doc/latex" 2>&1 | tee -a "${LOG_FILE}"
    #check_pipe_error
    #if test -f "${BUILD_DIR}/doc/latex/refman.pdf"; then
    #cp -v "${BUILD_DIR}/doc/latex/refman.pdf" "${TOP_DIR}/bml-manual.pdf"
    #fi
}

install() {
    ${CMAKE} --build "${BUILD_DIR}" --target install 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

testing() {
    cd "${BUILD_DIR}"
    ctest --verbose \
        --parallel ${PARALLEL_TEST_JOBS} \
        ${TESTING_EXTRA_ARGS} \
        2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error

    # Get skipped tests and re-run them with verbose output.
    local SKIPPED_TESTS
    readarray -t SKIPPED_TESTS < <(sed -E '0,/The following tests did not run:.*$/d; s/^\s*[0-9]+\s-\s([^ ]+) .*/\1/' "${LOG_FILE}")
    if (( ${#SKIPPED_TESTS[@]} > 0 )); then
        echo "Found skipped tests: ${SKIPPED_TESTS[*]}"
        local skipped
        for skipped in "${SKIPPED_TESTS[@]}"; do
            echo "Re-running skipped test ${skipped}"
            ctest --verbose \
                ${TESTING_EXTRA_ARGS} \
                --tests-regex "${skipped}" \
                2>&1 | tee -a "${LOG_FILE}"
        done
    fi
    cd "${TOP_DIR}"
}

check_indent() {
    cd "${TOP_DIR}"
    "${TOP_DIR}/scripts/indent.sh" 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

tags() {
    "${TOP_DIR}/scripts/update_tags.sh" 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

dist() {
    ${CMAKE} --build "${BUILD_DIR}" --target dist 2>&1 | tee -a "${LOG_FILE}"
    check_pipe_error
}

echo "Writing output to ${LOG_FILE}"

set_defaults

if [[ -n ${BLA_VENDOR} ]]; then
    if [[ -n ${BLAS_VENDOR} ]]; then
        echo "WARNING: BLAS_VENDOR (${BLAS_VENDOR}) will be used instead of BLA_VENDOR (${BLA_VENDOR})" | tee -a "${LOG_FILE}"
    else
        BLAS_VENDOR=${BLA_VENDOR}
    fi
fi

if [[ $# -gt 0 ]]; then
    if [[ $1 = "-h" || $1 = "--help" ]]; then
        help
        shift
    fi

    if [[ $# -gt 0 && $1 = "--debug" ]]; then
        PS4='+(${BASH_SOURCE##*/}:${LINENO}) ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
        set -x
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
                sanity_check
                configure
                ;;
            "compile")
                create
                sanity_check
                configure
                compile
                ;;
            "install")
                create
                sanity_check
                configure
                install
                ;;
            "testing")
                if [[ ${BML_TESTING} != "yes" ]]; then
                    echo "The testing step requires BML_TESTING to be true"
                    exit 1
                fi
                create
                sanity_check
                configure
                compile
                testing
                ;;
            "docs")
                create
                sanity_check
                configure
                docs
                ;;
            "indent")
                check_indent
                ;;
            "check_indent")
                check_indent
                ;;
            "tags")
                tags
                ;;
            "dist")
                create
                sanity_check
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
