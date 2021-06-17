#!/bin/bash

export BLAS_VENDOR=${BLAS_VENDOR:-Intel}
export BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-yes}
export COMMAND=${1:-compile}

./build.sh ${COMMAND}
