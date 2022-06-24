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


__global__ void dev_Hamiltonian(float* X, int N){
int j = threadIdx.x + blockIdx.x * blockDim.x; //row number
int i = blockIdx.y; //column number
if (i < N && j < N){

    if (i <= j){
        //printf("(%d,%d) \n",i,j);
        X[i*N+j] = exp(-0.5f*abs((float(i-j))))*sin(float(i+1));
        X[i+N*j] = X[i*N+j];
    }
}
}

void produce_hamiltonian (const unsigned N, float *X) {
    for(int i=0; i<N; ++i) {
        for(int j=i; j<N; ++j) {
            X[i+j*N] = exp(-0.5f*abs((float)(i-j)))*sin((float)(i+1));
            X[j+i*N] = X[i+j*N];
        }
    }
};

int main(int argc, char *argv[])
{

    // Matrix size
    size_t N = atoi(argv[1]);
    size_t Nocc = atoi(argv[2]);

     
    // Set GPU
    int device = 0;
    cudaSetDevice(device);

    // Cublas Handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Set math mode
    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    // Declare Memory,
    float  *sbuf1, *sbuf2, *S, *S2, *d_S, *d_S2;
    half   *hbuf1, *hbuf2;
    
    // Allocate host memory
    S = (float*) malloc(N * N * sizeof(float));
    S2 = (float*) malloc(N * N * sizeof(float));
    

    // Allocate device memory
    cudaMalloc(&d_S,N*N*sizeof(float));
    cudaMalloc(&d_S2,N*N*sizeof(float));

    // Allocate Buffers
    cudaMallocManaged(&sbuf1,  N * N * sizeof(float));
    cudaMallocManaged(&sbuf2,  N * N * sizeof(float));
    cudaMallocManaged(&hbuf1,  N * N * sizeof(half));
    cudaMallocManaged(&hbuf2,  N * N * sizeof(half));
    
    // Define grid size
    int numThreads = 128;
    int numBlocks = N * N / 80 / 128 + 1; 

    // Initialize Hamiltonian and identity
    produce_hamiltonian(N, S);
    cudaMemcpy(d_S, S, N*N*sizeof(float), cudaMemcpyHostToDevice);
    //dev_Hamiltonian<<<dim3 (N / 32 + 1, N, 1), dim3 (32, 1, 1)>>>(d_S,N);

    // Estimate spectral bounds using power method
    float h1 = -1.867;
    float hN = 1.867;
    
    // Get device id
    cudaGetDevice(&device); 
    
    //S^2 - half prec
    tcoretools::tcoreSPGemmSymm(handle
                                ,N
                                ,d_S
                                ,hbuf1
                                ,hbuf2
                                ,sbuf1
                                ,sbuf2
                                ,d_S2);

    
    //Deallocate device memory
    cudaFree(d_S);
    cudaFree(d_S2);
    cudaFree(sbuf1);
    cudaFree(sbuf2);
    cudaFree(hbuf1);
    cudaFree(hbuf2);

 
    // Destroy handle
    cublasDestroy(handle);

    return 0;
}



