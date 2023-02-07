#!/bin/bash

set -e -u -x

for READLINK in readlink greadlink; do
    if ${READLINK} --canonicalize ${HOME}; then
        break
    fi
done

basedir=$(${READLINK} --canonicalize $(dirname $0)/..)

export CC=${CC:-gcc-10}
export CXX=${CXX:-g++-10}
export FC=${FC:-gfortran-10}
export BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-no}
export BML_OPENMP=${BML_OPENMP:-no}
export BML_INTERNAL_BLAS=${BML_INTERNAL_BLAS:-no}
export TESTING_EXTRA_ARGS=${TESTING_EXTRA_ARGS:-"-R C-.*-single_real"}
export BML_VALGRIND=${BML_VALGRIND:-yes}

[[ -f ${basedir}/scripts/ci-defaults.sh ]] && . ${basedir}/scripts/ci-defaults.sh

${basedir}/build.sh --debug testing
