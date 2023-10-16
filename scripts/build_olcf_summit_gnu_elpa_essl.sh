#!/bin/bash
module load cmake
module load cuda
module load gcc/10.2.0
module load essl
module load magma
module load netlib-scalapack
module load netlib-lapack

rm -r build
rm -r install

MY_PATH=$(pwd)

#get jsrun with full path
JSRUN=$(which jsrun)
echo ${JSRUN}

export MAGMA_ROOT=${OLCF_MAGMA_ROOT:="${OLCF_MAGMA_ROOT}"}
export CC=${CC:=mpicc}
export FC=${FC:=mpif90}
export CXX=${CXX:=mpiCC}
export BML_OPENMP=${BML_OPENMP:=yes}
export BML_MPI=${BML_MPI:=yes}
export BML_OMP_OFFLOAD=${BML_OMP_OFFLOAD:=no}
export INSTALL_DIR=${INSTALL_DIR:="${MY_PATH}/install"}
export BML_TESTING=${BML_TESTING:=yes}
export BML_MAGMA=${BML_MAGMA:=yes}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:=Debug}

#set BLAS explicitly, otherwise cmake will pick the serial version of essl
export BLAS_LIBRARIES=${BLAS_LIBRARIES:="$OLCF_ESSL_ROOT/lib64/libesslsmp.so"}
#since essl does not contain all the lapack functions needed, we still need lapack
export LAPACK_LIBRARIES=${LAPACK_LIBRARIES:="$OLCF_NETLIB_LAPACK_ROOT/lib64/liblapack.so"}
export BML_SCALAPACK=${BML_SCALAPACK:=yes}
export SCALAPACK_LIBRARIES=${SCALAPACK_LIBRARIES:="-L$OLCF_NETLIB_SCALAPACK_ROOT/lib -lscalapack"}

export BML_CUDA=${BML_CUDA:=yes}
export BML_ELPA=${BML_ELPA:=yes}
export ELPA_DIR=${ELPA_DIR:=/ccs/proj/csc304/elpa/nvidia-gcc}
export EXTRA_LINK_FLAGS=${EXTRA_LINK_FLAGS:="-lgfortran"}

#use jsrun to run tests on a compute node
export BML_NONMPI_PRECOMMAND=${BML_NONMPI_PRECOMMAND:=${JSRUN}}
export BML_NONMPI_PRECOMMAND_ARGS=${BML_NONMPI_PRECOMMAND_ARGS:="-n1;-a1;-g1;-c7;--smpiargs=off"}

export BML_MPIEXEC_EXECUTABLE=${BML_MPIEXEC_EXECUTABLE:=${JSRUN}}
export BML_MPIEXEC_NUMPROCS_FLAG=${BML_MPIEXEC_NUMPROCS_FLAG:="-n"}
export BML_MPIEXEC_PREFLAGS=${BML_MPIEXEC_PREFLAGS:="-a1;-c4;-bpacked:2;-g1"}

./build.sh install
