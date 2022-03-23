#!/bin/bash

module load craype-accel-amd-gfx90a
module load rocm
module load cmake
export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

