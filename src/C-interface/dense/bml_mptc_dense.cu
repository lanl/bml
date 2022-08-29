#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>


// Copy from dense to already allocated bml_tc matrix type 
// if the matrix is not allocated in TC, then allocate and copy 

extern "C" void bml_mptc_dense(int, int, int, const double, double*, double*, int, const double, double*, int);

// Device function for splitting a single into two halves
__device__
void 
split_double(const double x, half &hi, half &lo)
{
    hi = __double2half(x);
    double y = (x - double(__half2float(hi)));
    lo = __double2half(y * 1024.0); // scale to maintain precision
};

template <typename T>
__global__
void 
array_split_double(const double *AD, T *AH1, T *AH2, const unsigned N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        half hi;
        half lo;

        split_double(AD[i], hi, lo);

        AH1[i] = hi;
        AH2[i] = lo;
    }
};

__global__
void 
array_float2double(const float *AF, double *AD, const unsigned N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        AD[i] = double(AF[i]);
    }
};


void 
tcoreDHgemm(cublasHandle_t &handle
           ,const unsigned M
	   ,const unsigned K
           ,const unsigned N
           ,const double alpha_
           ,const double* A_
           ,const double* B_
           ,half*  Ah
           ,half*  Al
           ,half*  Bh
           ,half*  Bl
           ,float* C
           ,const double beta_
           ,double* _C){

    cublasStatus_t cublasStat;

    // Setup kernel launch
    unsigned MAX_THREADS = 1024;
    unsigned BLOCKS_A = ceil(M * K / float(MAX_THREADS));
    unsigned BLOCKS_B = ceil(K * N / float(MAX_THREADS));
    unsigned BLOCKS_C = ceil(M * N / float(MAX_THREADS));
    unsigned THREADS = MAX_THREADS;

    // Split the double into the high and low parts
    array_split_double<half><<<BLOCKS_A, THREADS>>>(A_, Ah, Al, M*K);

    // Split the double into the high and low parts
    array_split_double<half><<<BLOCKS_B, THREADS>>>(B_, Bh, Bl, K*N);


    float gamma = powf(2,-10);
    float alpha = (float) alpha_ * gamma;
    float beta = (float) beta_ ;

    // Compute gemms for low, alpha*A_hi*B_lo + C = C 
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &alpha,
                              Ah, CUDA_R_16F, M,
                              Bl, CUDA_R_16F, K,
                              &beta, C, CUDA_R_32F, K, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    beta = 1.0f;
    // alpha*A_lo*B_hi + C = C
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &alpha,
                              Al, CUDA_R_16F, M,
                              Bh, CUDA_R_16F, K,
                              &beta, C, CUDA_R_32F, K, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    alpha = (float) alpha_ ;
    // Compute gemm for high, alpha*A_hi*B_hi + beta*C = C 
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &alpha,
                              Ah, CUDA_R_16F, M,
                              Bh, CUDA_R_16F, K,
                              &beta, C, CUDA_R_32F, K, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    /*// alpha*A_lo*B_hi + C = C
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              Al, CUDA_R_16F, N,
                              Bl, CUDA_R_16F, N,
                              &beta, C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    */
    // convert back to double-precision
    array_float2double<<<BLOCKS_C, THREADS>>>(C, _C, M*N);

};

void 
bml_mptc_dense(int M, int K, int N
              ,const double alpha
              ,double *A
              ,double *B, int ldb
              ,const double beta
              ,double *C, int ldc) 
{
    cudaError_t err;
    err = cudaSetDevice(0);

    // Cublas Handle
    cublasHandle_t handle;
    cublasStatus_t cublasStat = cublasCreate(&handle);

    // Set math mode
    cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
 
    // Define vars and allocate memory
    float *dev_C;
    half *dev_Ah, *dev_Al, *dev_Bh, *dev_Bl;
 
    cudaMalloc(&dev_C, M * N * sizeof(float));
    cudaMalloc(&dev_Ah, M * K * sizeof(half));
    cudaMalloc(&dev_Al, M * K * sizeof(half));
    cudaMalloc(&dev_Bh, K * N * sizeof(half));
    cudaMalloc(&dev_Bl, K * N * sizeof(half));

    // Do the multiply
    tcoreDHgemm(handle
               ,M, K, N
               ,alpha
               ,A
               ,B
               ,dev_Ah
               ,dev_Al
               ,dev_Bh
               ,dev_Bl
               ,dev_C
               ,beta
               ,C);

}



