#!/bin/bash

# Make sure all the paths are correct

rm -r build
#rm -r install_magma_2.7.2
rm -r install_hackathon

MY_PATH=$(pwd)


export CC=${CC:=cc}
export FC=${FC:=ftn}
export CXX=${CXX:=CC}
#export BLAS_VENDOR=${BLAS_VENDOR:=OpenBLAS}
#export LAPACK_LIBRARIES="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
#export BLAS_LIBRARIES="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
export LAPACK_LIBRARIES="-L/opt/cray/pe/libsci/23.12.5/GNU/12.3/aarch64/lib"
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_COMPLEX=${BML_COMPLEX:=yes}
export BML_CUSOLVER=${BML_CUSOLVER:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install_hackathon"}
#export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export MAGMA_ROOT=${MAGMA_ROOT:="/usr/projects/icapt/mewall/venado/packages/magma-2.7.2/install"}
export BML_MAGMA=${BML_MAGMA:=yes}
#export BML_POSIX_MEMALIGN=${BML_POSIX_MEMALIGN:=no}
export CMAKE_PREFIX_PATH=$CUDA_HOME/../../math_libs/$CRAY_CUDATOOLKIT_VERSION:$CMAKE_PREFIX_PATH
#export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-DCRAY_SDK -I${CUDA_ROOT}/include -lcudart"}
#export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-DCRAY_SDK -L${CUDA_ROOT}/lib64 -lcudart"}

./build.sh configure


