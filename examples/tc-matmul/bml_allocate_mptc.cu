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


bml_zero_matrix_dense_mptc (N)
     int main (int argc, char *argv[])
{


  // Set GPU
  int device = 0;
  cudaSetDevice (device);

  // Declare Memory,
  float /* *sbuf1, *sbuf2,*/ *S, *S2, *d_S, *d_S2;
  //half *hbuf1, *hbuf2;

  // Allocate host memory
  S = (float *) malloc (N * N * sizeof (float));
  S2 = (float *) malloc (N * N * sizeof (float));


  // Allocate device memory
  cudaMalloc (&d_S, N * N * sizeof (float));
  cudaMalloc (&d_S2, N * N * sizeof (float));

  // Allocate Buffers
  //cudaMallocManaged (&sbuf1, N * N * sizeof (float));
  //cudaMallocManaged (&sbuf2, N * N * sizeof (float));
  //cudaMallocManaged (&hbuf1, N * N * sizeof (half));
  //cudaMallocManaged (&hbuf2, N * N * sizeof (half));


  return 0;
}
