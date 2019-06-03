#!/bin/bash

# Make sure all the paths are correct

rm -r build.broadwell
rm -r install.broadwell

MY_PATH=$(pwd)

export CC=${CC:=gcc}
export FC=${FC:=gfortran}
export CXX=${CXX:=g++}
export BLAS_VENDOR=${BLAS_VENDOR:=GNU}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}

./build.sh configure                                                                                                                                                                                              
                                    
