#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Copy from dense to already allocated bml_tc matrix type 
// if the matrix is not allocated in TC, then allocate and copy 

extern "C" void bml_mptc_dense(int, double, double*, double*, double, double*);

// Device function for splitting a single into two halves
__device__
void 
split_single(const float x, half &hi, half &lo)
{
    hi = __float2half(x);
    float y = (x - __half2float(hi));
    lo = __float2half(y * 1024.0);
};

template <typename T>
__global__
void 
array_split_single(const float *AF, T *AH1, T *AH2, const unsigned N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        half hi;
        half lo;

        split_single(AF[i], hi, lo);

        AH1[i] = hi;
        AH2[i] = lo;
    }
};

void 
tcoreSPGemmSymm(cublasHandle_t &handle
               ,const unsigned N
               ,const float* A
               ,const float* B
               ,half*  Ah
               ,half*  Al
               ,half*  Bh
               ,half*  Bl
               ,float* C1
               ,float* C2
               ,float* C){
//               ,cudaStream_t cuStrm) {

    cublasStatus_t cublasStat;

    // Setup kernel launch
    unsigned MAX_THREADS = 1024;
    unsigned BLOCKS = ceil(N*N/float(MAX_THREADS));
    unsigned THREADS = MAX_THREADS;

    // Split the floats into the high and low parts
    array_split_single<half><<<BLOCKS, THREADS>>>(A, Ah, Al, N*N);

    // Split the floats into the high and low parts
    array_split_single<half><<<BLOCKS, THREADS>>>(B, Bh, Bl, N*N);
    
    // Set the math mode to allow cuBLAS to use Tensor Cores:
    //cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha (1.0f);
    float beta  (0.0f);
    float gamma = powf(2,-10);


    // Compute gemm for high
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Ah, CUDA_R_16F, N,
                              Bh, CUDA_R_16F, N,
                              &beta, C1, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Compute gemms for low
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Ah, CUDA_R_16F, N,
                              Bl, CUDA_R_16F, N,
                              &beta, C2, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Al, CUDA_R_16F, N,
                              Bh, CUDA_R_16F, N,
                              &alpha, C2, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // add the high gemm and low gemm together
    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N,
                             &alpha,
                             C1, N,
                             &gamma,
                             C2, N,
                             C2, N);

    // compute C + C^T 
    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             N, N,
                             &alpha,
                             C2, N,
                             &alpha,
                             C2, N,
                             C, N);  

};

void 
bml_mptc_dense(int N, double alpha, double *A, double *B, double beta, double *C) 
//double alpha, double *A, double *B, double beta, double *C, int ldc)
{
    cudaError_t err;
    err = cudaSetDevice(0);
    printf("cuda return value = %d \n", err);

    // Cublas Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set math mode
    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

 
    // Define vars and allocate memory
    float *dev_A, *dev_B, *dev_C, *dev_C1, *dev_C2;
    half *dev_Ah, *dev_Al, *dev_Bh, *dev_Bl;
 
    cudaMalloc(&dev_A,  N * N * sizeof(float));
    cudaMalloc(&dev_B,  N * N * sizeof(float));
    cudaMalloc(&dev_C,  N * N * sizeof(float));
    cudaMalloc(&dev_C1, N * N * sizeof(float));
    cudaMalloc(&dev_C2, N * N * sizeof(float));
    cudaMalloc(&dev_Ah, N * N * sizeof(half));
    cudaMalloc(&dev_Al, N * N * sizeof(half));
    cudaMalloc(&dev_Bh, N * N * sizeof(half));
    cudaMalloc(&dev_Bl, N * N * sizeof(half));



    // Do the multiply
    tcoreSPGemmSymm(handle
                   ,N
                   ,dev_A
                   ,dev_B
                   ,dev_Ah
                   ,dev_Al
                   ,dev_Bh
                   ,dev_Bl
                   ,dev_C1
                   ,dev_C2
                   ,dev_C);

}









