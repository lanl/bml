/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright 2010.  Los Alamos National Security, LLC. This material was    !
! produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos !
! National Laboratory (LANL), which is operated by Los Alamos National     !
! Security, LLC for the U.S. Department of Energy. The U.S. Government has !
! rights to use, reproduce, and distribute this software.  NEITHER THE     !
! GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY,     !
! EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS         !
! SOFTWARE.  If software is modified to produce derivative works, such     !
! modified software should be clearly marked, so as not to confuse it      !
! with the version available from LANL.                                    !
!                                                                          !
! Additionally, this program is free software; you can redistribute it     !
! and/or modify it under the terms of the GNU General Public License as    !
! published by the Free Software Foundation; version 2.0 of the License.   !
! Accordingly, this program is distributed in the hope that it will be     !
! useful, but WITHOUT ANY WARRANTY; without even the implied warranty of   !
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General !
! Public License for more details.                                         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include "Matrix.h"

extern int ndevices;
extern int nblocks;

// Initialize sparse matrix in csr format
void M_InitSparseMatrix(SparseMatrix &A, int N, int msparse) {

  int Npad;
  int cdev;

  // Get current device
  cudaGetDevice(&cdev);

  A.HDIM = N;
  A.M=N;
  A.N=N;
  A.K=N;
  A.MSPARSE=msparse;

  /* If we have 1 block we can use our optimal padding scheme. Also, since 
     our optimal paddings are always divisible by 2, we can use them when 
     we run with 2 blocks. On the other hand, if there's more than two it is 
     essential to make sure the dimensions are an integer multiple of the 
     number of blocks */
  
  if ( nblocks <= 2 ) {
    
    if (sizeof(REAL) == 8) {
      
      // double precision case - multiples of 64 best
      
      if (N <= 736) {
	
	//Npad = N;
	
	Npad = nblocks*((N-1)/nblocks + 1);
	
      } else if ( N > 736 ) {
	
	Npad = 16*((N-1)/16 + 1);
	
      } 
      
    }
    
    if (sizeof(REAL) == 4) {
      
      // Single precision dimensions
      
      if (N <= 448) {
	
	Npad = nblocks*((N-1)/nblocks + 1);
	
      } else if ( N > 448 ) {
	
	Npad = 16*((N - 1)/16 + 1);
	
      }
      
    }
    
  } else if ( nblocks > 2 ) {
    
    Npad = nblocks*((N - 1)/nblocks + 1);
    
    //    printf("%d %d %d\n", nblocks, N, Npad);
    
  }
  
  A.N = Npad;
  A.M = A.N*A.M/A.N;

//  printf("Init: %d %d %d %d %d \n", M, N, Npad, A.DM, A.DN);  

  // Sparse Matrix in csr format on GPUs
  int msize = A.N * A.MSPARSE;
  for (int d = 0; d < ndevices; d++) {
    cudaSetDevice(d);

    // Initialize descriptor
    cusparseCreateMatDescr(&A.descr);
    cusparseSetMatDiagType(A.descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(A.descr, CUSPARSE_INDEX_BASE_ZERO);
    //cusparseSetMatFillMode(A.descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatType(A.descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    // Number of non-zeroes pointer
    A.nnzTotalDevHostPtr = &A.nnz;

    // Row index pointers
    cudaMalloc((void **)&A.csrRowPtr, (A.N+1)*sizeof(int));
    cudaMemset(A.csrRowPtr, '\0', (A.N+1)*sizeof(int));

    // Actual column indeces
    cudaMalloc((void **)&A.csrColInd, msize*sizeof(int));
    cudaMemset(A.csrColInd, '\0', msize*sizeof(int));

    // Non-zero values
    cudaMalloc((void **)&A.csrVal, msize*sizeof(REAL));
    cudaMemset(A.csrVal, '\0', msize*sizeof(REAL));  
  }

  // Restore device
  cudaSetDevice(cdev);

}
