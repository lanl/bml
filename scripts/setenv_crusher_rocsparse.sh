#!/bin/bash

module swap cce cce/14.0.2
module load craype-accel-amd-gfx90a
module load rocm/5.2.0
module load cmake
module load openblas
