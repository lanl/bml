#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <random>
#include <ctime>

#include "tcore_hp_emulator.cuh"

// Device function for splitting a single into two halves
__device__
void split_single(const float x, half &hi, half &lo)
{
    hi = __float2half(x);
    float y = (x - __half2float(hi));
    lo = __float2half(y * 1024.0);
}

template <typename T>
__global__
void array_split_single(const float *AF, T *AH1, T *AH2, const unsigned N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        half hi;
        half lo;

        split_single(AF[i], hi, lo);

        //AH1[i] = __half2float(hi);
        //AH2[i] = __half2float(lo);
        AH1[i] = hi;
        AH2[i] = lo;
    }
}

void tcoretools::tcoreSPGemmSymm (cublasHandle_t &handle
                                 ,const unsigned N
                                 ,const float* A
                                 ,half*  Ah
                                 ,half*  Al
                                 ,float* B1
                                 ,float* B2
                                 ,float* B
                                 ,cudaStream_t cuStrm) {
    // Setup kernel launch
    unsigned MAX_THREADS = 1024;
    unsigned BLOCKS = ceil(N*N/float(MAX_THREADS));
    unsigned THREADS = MAX_THREADS;

    // Split the floats into the high and low parts
    array_split_single<half><<<BLOCKS, THREADS>>>(A, Ah, Al, N*N);

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha (1.0f);
    float beta  (0.0f);

    // Compute gemm for high
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Ah, CUDA_R_16F, N,
                              Ah, CUDA_R_16F, N,
                              &beta, B1, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Compute gemm for low
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Ah, CUDA_R_16F, N,
                              Al, CUDA_R_16F, N,
                              &beta, B2, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    alpha = 1.0f;
    beta = 1.0f;
    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             N, N,
                             &alpha,
                             B2, N,
                             &beta,
                             B2, N,
                             B, N);

    beta = powf(2,-10);
    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N,
                             &alpha,
                             B1, N,
                             &beta,
                             B, N,
                             B, N);
};

void tcoretools::tcoreSPGemmSP2iter (cublasHandle_t &handle
                                    ,const unsigned N
                                    ,const float* A
                                    ,half*  Ah
                                    ,half*  Al
                                    ,float* B1
                                    ,float* B2
                                    ,float* B
                                    ,cudaStream_t cuStrm) {
    // Setup kernel launch
    unsigned MAX_THREADS = 1024;
    unsigned BLOCKS = ceil(N*N/float(MAX_THREADS));
    unsigned THREADS = MAX_THREADS;

    // Split the floats into the high and low parts
    array_split_single<half><<<BLOCKS, THREADS>>>(A, Ah, Al, N*N);

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha (1.0f);
    float beta  (0.0f);

    // Compute gemm for high
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Ah, CUDA_R_16F, N,
                              Ah, CUDA_R_16F, N,
                              &beta, B1, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Compute gemm for low
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Ah, CUDA_R_16F, N,
                              Al, CUDA_R_16F, N,
                              &beta, B2, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    alpha = 1.0f;
    beta = 1.0f;
    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             N, N,
                             &alpha,
                             B2, N,
                             &beta,
                             B2, N,
                             B, N);

    beta = powf(2,-10);
    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N,
                             &alpha,
                             B1, N,
                             &beta,
                             B, N,
                             B, N);
};
