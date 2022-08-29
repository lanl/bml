#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>


// Copy from dense to already allocated bml_tc matrix type 
// if the matrix is not allocated in TC, then allocate and copy 

extern "C" void bml_mptc_symm_x2_dense(int, double*, double*);

// Device function for splitting a single into two halves
__device__
void 
split_double_x2(const double x, half &hi, half &lo)
{
    hi = __double2half(x);
    double y = (x - double(__half2float(hi)));
    lo = __double2half(y * 1024.0); // scale to maintain precision
};

template <typename T>
__global__
void 
array_split_double_x2(const double *AD, T *AH1, T *AH2, const unsigned N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        half hi;
        half lo;

        split_double_x2(AD[i], hi, lo);

        AH1[i] = hi;
        AH2[i] = lo;
    }
};

__global__
void 
array_float2double_x2(const float *AF, double *AD, const unsigned N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        AD[i] = double(AF[i]);
    }
};


void 
tcoreDHSymGemm(cublasHandle_t &handle
              ,const unsigned N
              ,const double* A_
              ,half*  Ah
              ,half*  Al
              ,float* C
              ,float* temp
              ,double* _C){

    cublasStatus_t cublasStat;

    // Setup kernel launch
    unsigned MAX_THREADS = 1024;
    unsigned BLOCKS = ceil(N * N / float(MAX_THREADS));
    unsigned THREADS = MAX_THREADS;

    // Split the double into the high and low parts
    array_split_double_x2<half><<<BLOCKS, THREADS>>>(A_, Ah, Al, N * N);

    float alpha = (float) powf(2,-10);
    float beta = (float) 0.0 ;
    float gamma = (float) 1.0 ;

    // Compute gemm for high*low, alpha*A_hi*A_lo + 0.0*C = C 
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                              N, N, N, 
                              &alpha,
                              Ah, CUDA_R_16F, N,
                              Al, CUDA_R_16F, N,
                              &beta, 
                              C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Add transpose of C to C, C + (A_hi*A_lo)^t = C 
    cublasStat = cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             N, N,
                             &gamma,
                             C, N,
                             &gamma,
                             C, N,
                             C, N);

    // Compute gemm for high, A_hi*B_hi + C = C 
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                              N, N, N, 
                              &gamma,
                              Ah, CUDA_R_16F, N,
                              Ah, CUDA_R_16F, N,
                              &gamma, 
                              C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // convert C back to double-precision
    array_float2double_x2<<<BLOCKS, THREADS>>>(C, _C, N * N);

};

void 
bml_mptc_symm_x2_dense(int N
                      ,double *A
                      ,double *A2) 
{
    cudaError_t err;
    err = cudaSetDevice(0);

    // Cublas Handle
    cublasHandle_t handle;
    cublasStatus_t cublasStat = cublasCreate(&handle);

    // Set math mode
    cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
 
    // Define vars and allocate memory
    float *dev_A2, *dev_temp;
    half *dev_Ah, *dev_Al;
 
    cudaMalloc(&dev_A2, N * N * sizeof(float));
    cudaMalloc(&dev_temp, N * N * sizeof(float));
    cudaMalloc(&dev_Ah, N * N * sizeof(half));
    cudaMalloc(&dev_Al, N * N * sizeof(half));

    // Do the square with symmetric matrices
    tcoreDHSymGemm(handle
                  ,N
                  ,A
                  ,dev_Ah
                  ,dev_Al
                  ,dev_A2
                  ,dev_temp
                  ,A2);

}



