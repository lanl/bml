#!/bin/bash

# Make sure all the paths are correct

source setenv_lassen_offload.sh

rm -r build
rm -r install

MY_PATH=$(pwd)

export CC=${CC:=xlc-gpu}
export FC=${FC:=xlf2003-gpu}
export CXX=${CXX:=xlc++-gpu}
export BLAS_VENDOR=${BLAS_VENDOR:=Auto}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_OMP_OFFLOAD=${BML_OMP_OFFLOAD:=yes}
export BML_CUSPARSE=${BML_CUSPARSE:=yes}
export BML_COMPLEX=${BML_COMPLEX:=no}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:=""}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:=""}
#export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-lm -L/usr/tce/packages/xl/xl-2021.03.11/xlC/16.1.1/lib -libmc++"}
export BLAS_LIBRARIES=${BLAS_LIBRARIES:="-L${LAPACK_DIR} -llapack -lblas"}

./build.sh configure

pushd build
make -j 
make install
popd


                                                                                                                                                                                              
                                    
