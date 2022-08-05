#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Copy from dense to already allocated bml_tc matrix type 
// if the matrix is not allocated in TC, then allocate and copy 

void 
bml_mptc_dense (double *A, double *B, double *C)
{
  cudaError_t err;

  err = cudaSetDevice(0);




  

  //if (err == 0){
      printf("BERGA BERG");
 // }

}



// Device function for splitting a single into two halves
__device__
void split_single(const float x, half &hi, half &lo)
{
    hi = __float2half(x);
    float y = (x - __half2float(hi));
    lo = __float2half(y * 1024.0);
}

/*template <typename T>
__global__
void array_split_single(const float *AF, T *AH1, T *AH2, const unsigned N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        half hi;
        half lo;

        split_single(AF[i], hi, lo);

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
*/
