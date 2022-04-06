#!/bin/bash

#module purge
module load cmake
module load xl/2021.03.11-cuda-11.2.0
module load cuda/11.2.0
module load lapack/3.9.0-xl-2020.11.12
#module load essl
export CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR="/usr/tce/packages/cuda/cuda-11.2.0"}

