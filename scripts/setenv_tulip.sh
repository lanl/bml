#!/bin/bash

module load PrgEnv-cray
module load cce/11.0.3
module load rocm/4.1.0
module use /home/groups/coegroup/share/coe/modulefiles
module load hipmagma
module load cmake
export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

