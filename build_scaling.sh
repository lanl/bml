#!/bin/bash

: ${BLAS_VENDOR:=Intel}
: ${BUILD_SHARED_LIBS:=yes}

export BLAS_VENDOR
export BUILD_SHARED_LIBS

./build.sh install
