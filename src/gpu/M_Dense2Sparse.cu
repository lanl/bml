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

extern cusparseHandle_t **cshandle;
extern int ndevices;
extern int nblocks;

// Convert dense matrix to sparse matrix in csr format
void M_Dense2Sparse(Matrix &A, SparseMatrix &B) {

  int* nnzPerRow;
  int cdev;

  // Get current device
  cudaGetDevice(&cdev);

  // On each GPU
  for (int d = 0; d < ndevices; d++) {

    cudaSetDevice(d);

    // Allocate space for array of number of non-zeros per row
    cudaMalloc((void **)&nnzPerRow, A.DN*sizeof(int));

    // Determine number of non-zeros in x0
#if REALSIZE==4
    cusparseSnnz(cshandle[d][0], CUSPARSE_DIRECTION_ROW, A.DN, A.DM,
      B.descr, A.Device[d], A.DN, nnzPerRow, B.nnzTotalDevHostPtr);
#elif REALSIZE==8
    cusparseDnnz(cshandle[d][0], CUSPARSE_DIRECTION_ROW, A.DN, A.DM,
      B.descr, A.Device[d], A.DN, nnzPerRow, B.nnzTotalDevHostPtr);
#endif

    B.nnz = *(B.nnzTotalDevHostPtr);

    // Convert dense to sparse csr format
#if REALSIZE==4
    cusparseSdense2csr(cshandle[d][0], B.N, B.M, B.descr,
      A.Device[d], A.DN, nnzPerRow, B.csrVal, B.csrRowPtr, B.csrColInd);
#elif REALSIZE==8
    cusparseDdense2csr(cshandle[d][0], B.N, B.M, B.descr,
      A.Device[d], A.DN, nnzPerRow, B.csrVal, B.csrRowPtr, B.csrColInd);
#endif

    cudaFree(nnzPerRow);

  }

  // Restore device
  cudaSetDevice(cdev);

}
