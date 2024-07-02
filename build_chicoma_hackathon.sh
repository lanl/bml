#!/bin/bash

# Make sure all the paths are correct

rm -r build
#rm -r install_magma_2.7.2
rm -r install_hackathon

MY_PATH=$(pwd)


export CC=${CC:=cc}
export FC=${FC:=ftn}
export CXX=${CXX:=CC}
#export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-ef -DCRAY_SDK"}
#export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-DINTEL_SDK"}
#export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-DCRAY_SDK"}
export BLAS_VENDOR=${BLAS_VENDOR:=Intel10_64lp}
#export BLAS_VENDOR=${BLAS_VENDOR:=OpenBLAS}
#export LAPACK_LIBRARIES="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
#export BLAS_LIBRARIES="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_COMPLEX=${BML_COMPLEX:=yes}
#export CUDA_ROOT=${CUDA_PATH}
#export CUDA_DIR=${CUDA_PATH}
#export CUDA_INCLUDE=${CUDA_INCLUDES}
#CUDA_INCLUDESexport CUDA_LIB=${CUDA_LIBS}
#export CUDA_TOOLKIT_ROOT_DIR=${CUDATOOLKIT_ROOT}
#export BML_CUDA=${BML_CUDA:=yes}
export BML_CUSOLVER=${BML_CUSOLVER:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install_hackathon"}
#export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
#export MAGMA_ROOT=${MAGMA_ROOT:="/usr/projects/icapt/mewall/packages/gpmd/magma/install"}
export MAGMA_ROOT=${MAGMA_ROOT:="/usr/projects/icapt/mewall/packages/gpmd/magma-2.7.2/install"}
export BML_MAGMA=${BML_MAGMA:=yes}
#export BML_POSIX_MEMALIGN=${BML_POSIX_MEMALIGN:=no}
#export CMAKE_PREFIX_PATH="$MKLROOT/lib/intel64"
#export EXTRA_FFLAGS=${EXTRA_FFLAGS:="-fdefault-integer-8"}
#export EXTRA_CFLAGS=${EXTRA_CFLAGS:=""}
#export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-DCRAY_SDK -I${CUDA_ROOT}/include -lcudart"}
#export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-DCRAY_SDK -L${CUDA_ROOT}/lib64 -lcudart"}

./build.sh configure


