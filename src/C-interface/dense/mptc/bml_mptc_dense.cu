#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Copy from dense to already allocated bml_tc matrix type 
// if the matrix is not allocated in TC, then allocate and copy 

void 
bml_mptc_dense ()
{
  cudaError_t err;

  err = cudaSetDevice(0);
  

  /*if (err == 0){
      printf("BERG BERG");
  }*/
  exit(0);  
}
