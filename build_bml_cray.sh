#!/bin/bash

# Make sure all the paths are correct

rm -rf build
rm -rf install_cray

MY_PATH=$(pwd)


export CC=${CC:=cc}
#export FC=${FC:=/opt/cray/pe/cce/17.0.0/bin/crayftn}
export FC=${FC:=ftn}
export CXX=${CXX:=CC}
#export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-ef -DCRAY_SDK"}
#export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-DINTEL_SDK"}
#export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-DCRAY_SDK"}
#export BLAS_VENDOR=${BLAS_VENDOR:=OpenBlas}
#export BLAS_VENDOR=${BLAS_VENDOR:=Intel10_64lp}
#export BLAS_LIBRARIES="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
#export BLAS_LIBRARIES="-L/projects/darwin-nv/rhel9/aarch64/packages/nvpl/23.11/lib -lnvpl_blas_lp64_gomp"
#export LAPACK_LIBRARIES="-L/projects/darwin-nv/rhel9/aarch64/packages/nvpl/23.11/lib -lnvpl_lapack_lp64_gomp"
export LAPACK_LIBRARIES="-L${CRAY_PE_LIBSCI_PREFIX}/lib"
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_COMPLEX=${BML_COMPLEX:=no}
#export CUDA_ROOT=${CUDA_PATH}
#export CUDA_DIR=${CUDA_PATH}
#export CUDA_INCLUDE=${CUDA_INCLUDES}
#CUDA_INCLUDESexport CUDA_LIB=${CUDA_LIBS}
#export CUDA_TOOLKIT_ROOT_DIR=${CUDATOOLKIT_ROOT}
#export BML_CUDA=${BML_CUDA:=yes}
export BML_CUSOLVER=${BML_CUSOLVER:=yes}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install_cray"}
#export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export MAGMA_ROOT=${MAGMA_ROOT:="/usr/projects/icapt/mewall/venado/packages/magma-2.7.2/install_cray"}
export BML_MAGMA=${BML_MAGMA:=yes}
export CMAKE_PREFIX_PATH="$CUDATOOLKIT_HOME/../../math_libs/lib64"
#export EXTRA_FFLAGS=${EXTRA_FFLAGS:="-DNVHPC_SDK"}
export CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS:="-DCRAY_SDK -ef"}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:=""}
export BML_POSIX_MEMALIGN=${BML_POSIX_MEMALIGN:=no}
#export EXTRA_CFLAGS=${EXTRA_CFLAGS:="-DCRAY_SDK -I${CUDA_ROOT}/include -lcudart"}
#export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-DCRAY_SDK -L${CUDA_ROOT}/lib64 -lcudart"}

./build.sh configure


