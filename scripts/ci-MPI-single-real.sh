#!/bin/bash

set -e -u -x

basedir=$(readlink --canonicalize $(dirname $0)/..)

export CC=${CC:-mpicc}
export CXX=${CXX:-mpic++}
export FC=${FC:-mpifort}
export BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-no}
export BML_OPENMP=${BML_OPENMP:-no}
export BML_INTERNAL_BLAS=${BML_INTERNAL_BLAS:-no}
export BML_MPI=${BML_MPI:-yes}
export TESTING_EXTRA_ARGS=${TESTING_EXTRA_ARGS:-"-R MPI-C-.*-single_real"}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:-"-lscalapack-openmpi"}
export BML_SCALAPACK=${BML_SCALAPACK:-yes}

[[ -f ${basedir}/scripts/ci-defaults.sh ]] && . ${basedir}/scripts/ci-defaults.sh

${basedir}/build.sh --debug testing
