#!/bin/bash
# Make sure all the paths are correct

#BASE=/home/ghadar/EXAALT/LATTE_Nov_11_2020/bml
#BUILD=${BASE}/build_icpx_cpu
#INSTALL=${BASE}/install_icpx_cpu
#rm -r ${BUILD}
#rm -r ${INSTALL}

rm -r build
rm -r install

MY_PATH=$(pwd)

export CC=icx
export FC=ifx
export CXX=icpx
export MKL_GPU=yes

export BLAS_VENDOR=${BLAS_VENDOR:=MKL}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_CUDA=${BML_CUDA:=no}
export MKL_GPU=${MKL_GPU:=yes}
export BML_OMP_OFFLOAD=${BML_OMP_OFFLOAD:=no}
export BML_MAGMA=${BML_MAGMA:=no}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BUILD_DIR=${BUILD_DIR:="${MY_PATH}/build"}
export BML_TESTING=${BML_TESTING:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Release}
export EXTRA_CFLAGS=${EXTRA_CFLAGS:=""}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:=""}
export CUDA_TOOLKIT_ROOT_DIR=""
export CMAKE_Fortran_FLAGS="-fiopenmp -fopenmp-targets=spir64 -DINTEL_SDK"
export CMAKE_C_FLAGS="-O3 -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ -DINTEL_SDK"

export EXTRA_LINK_FLAGS="-L/soft/restricted/CNDA/sdk/2021.04.30.001/oneapi/mkl/2021u3_20210525/lib/intel64 -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -lmkl_intel_ilp64  -lmkl_sycl -lstdc++ -Wl,-rpath,/soft/restricted/CNDA/sdk/2021.04.30.001/oneapi/kokkos/20210323-3.1/../../compiler/latest/linux/compiler/lib/intel64 -L/soft/restricted/CNDA/sdk/2021.04.30.001/oneapi/kokkos/20210323-3.1/../../compiler/latest/linux/compiler/lib/intel64 -liomp5 -lsycl -lOpenCL -lm -lpthread -ldl"

 

./build.sh configure
#./build.sh install
cd build
make -j  VERBOSE=1
make test
make install

cd ../install
ln -s lib64 lib

