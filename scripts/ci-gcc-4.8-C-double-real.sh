#!/bin/bash

set -e -u -x

basedir=$(readlink --canonicalize $(dirname $0)/..)

export CC=${CC:-gcc-4.8}
export CXX=${CXX:-g++-4.8}
export FC=${FC:-gfortran-4.8}
export BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-no}
export BML_OPENMP=${BML_OPENMP:-no}
export BML_INTERNAL_BLAS=${BML_INTERNAL_BLAS:-no}
export TESTING_EXTRA_ARGS=${TESTING_EXTRA_ARGS:-"-R C-.*-double_real"}
export BML_VALGRIND=${BML_VALGRIND:-yes}

[[ -f ${basedir}/scripts/ci-defaults.sh ]] && . ${basedir}/scripts/ci-defaults.sh

${basedir}/build.sh --debug testing
