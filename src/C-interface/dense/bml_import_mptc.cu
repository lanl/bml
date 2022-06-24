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

// Copy from dense to already allocated bml_tc matrix type 

// if the matrix is not allocated in TC, the allocate and copy 

extern "C"
{
    float *bml_import_mptc (float *);
}

float *
bml_import_mptc (float *A)
{
  int N = 100;
  // Set GPU
  int device = 0;
  cudaSetDevice (device);

  // Cublas Handle
  cublasHandle_t handle;
  cublasCreate (&handle);

  // Set math mode
  cublasStatus_t cublasStat =
    cublasSetMathMode (handle, CUBLAS_TENSOR_OP_MATH);

  float * A_bml;
  A_bml = (float*)malloc(N * N * sizeof(float));
  printf("bergaberg");
  cudaMemcpy (A_bml, A, N * N * sizeof (float), cudaMemcpyHostToDevice);

  return A_bml;
}
