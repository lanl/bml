#!/bin/bash
module load gcc/5.4.0
module load netlib-lapack/3.6.1
module load magma/2.2.0
module load cuda/8.0.54
module load cmake/3.6.1

# Make sure all the paths are correct

rm -r build
rm -r install

MY_PATH=$(pwd)

export BML_OPENMP=${BML_OPENMP:=no}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_MAGMA=${BML_MAGMA:=yes}
export MAGMA_ROOT=$OLCF_MAGMA_ROOT

./build.sh testing
#./build.sh install
                                                                                                                                                                                              
                                    
