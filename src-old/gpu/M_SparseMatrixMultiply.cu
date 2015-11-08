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

extern cusparseHandle_t* cshandle;
extern int ndevices;
extern int nblocks;

// Multiply matrices in csr format - A * B = C
void M_SparseMatrixMultiply(SparseMatrix &A, SparseMatrix &B, SparseMatrix &C) {

  int cdev;

  // Get current device
  cudaGetDevice(&cdev);

  // On each GPU
  for (int d = 0; d < ndevices; d++) {

    cudaSetDevice(d);

    // Determine number of non-zeros for multiplication - C = A * B
    cusparseXcsrgemmNnz(cshandle[d], 
        CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, A.N, A.M, A.K,
        A.descr, A.nnz, A.csrRowPtr, A.csrColInd,
        B.descr, B.nnz, B.csrRowPtr, B.csrColInd,
        C.descr, C.csrRowPtr, C.nnzTotalDevHostPtr);

    C.nnz = *C.nnzTotalDevHostPtr;

#if REALSIZE==4
    cusparseScsrgemm(cshandle[d],
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, A.N, A.M, A.K,
        A.descr, A.nnz,
        A.csrVal, A.csrRowPtr, A.csrColInd,
        B.descr, B.nnz,
        B.csrVal, B.csrRowPtr, B.csrColInd,
        C.descr,
        C.csrVal, C.csrRowPtr, C.csrColInd);
#elif REALSIZE==8
    cusparseDcsrgemm(cshandle[d], 
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, A.N, A.M, A.K,
        A.descr, A.nnz,
        A.csrVal, A.csrRowPtr, A.csrColInd,
        B.descr, B.nnz,
        B.csrVal, B.csrRowPtr, B.csrColInd,
        C.descr,
        C.csrVal, C.csrRowPtr, C.csrColInd);
#endif
  }

  // Restore device
  cudaSetDevice(cdev);

}
