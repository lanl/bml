// Create a zero matrix in GPU 

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <regex>
#include <typeinfo>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include <vector>
#include "tcore_hp_emulator.cuh"



int bml_deallocate_mptc (cublasHandle_t handle, float * A)
{


  cublasDestroy(handle);
  cudaFree(A);
  
  return 0;
};
