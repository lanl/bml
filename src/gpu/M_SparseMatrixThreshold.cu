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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/gather.h>

#include <iostream>

extern int ndevices;
extern int nblocks;
extern cudaStream_t **stream;
extern cudaEvent_t **event;

  // Check for non-zeroes
  struct nonZero
  {
    nonZero(REAL thr) : threshold(thr) {}
    __host__ __device__
    __device__ int operator()(const REAL x)
    {
      return ((abs(x) >= threshold) ? 1 : 0);
    }
    const REAL threshold;
  };

  // Check for ones
  struct isOne
  {
    __host__ __device__
    __device__ int operator()(const int x)
    {
      return (x == 1);
    }
  };
  
// Threshold A to get B
void M_SparseMatrixThreshold(REAL numthresh, SparseMatrix &A, SparseMatrix &B) {

  // Make arrays device_ptr's for use with thrust algorithms
  thrust::device_ptr<int> row_ptr = thrust::device_pointer_cast(A.csrRowPtr);
  thrust::device_ptr<int> col_ptr = thrust::device_pointer_cast(A.csrColInd);
  thrust::device_ptr<REAL> val_ptr = thrust::device_pointer_cast(A.csrVal);

  thrust::device_ptr<int> row2_ptr = thrust::device_pointer_cast(B.csrRowPtr);
  thrust::device_ptr<int> col2_ptr = thrust::device_pointer_cast(B.csrColInd);
  thrust::device_ptr<REAL> val2_ptr = thrust::device_pointer_cast(B.csrVal);

  thrust::device_ptr<int> flag_ptr = thrust::device_malloc<int>(A.nnz);


  // Flag all non-zero values with 1
  thrust::transform(val_ptr, val_ptr+A.nnz, flag_ptr, nonZero(numthresh));

  // Count number of non-zeros
  B.nnz = row2_ptr[A.HDIM] = thrust::count(flag_ptr, flag_ptr+A.nnz, (int)1);

  // Compact non-zero values and indeces - stream compaction
  thrust::copy_if(
    thrust::make_zip_iterator(thrust::make_tuple(val_ptr, col_ptr)),
    thrust::make_zip_iterator(thrust::make_tuple(val_ptr+A.nnz, col_ptr+A.nnz)),
    flag_ptr, 
    thrust::make_zip_iterator(thrust::make_tuple(val2_ptr, col2_ptr)),
    isOne());

  // In-place prefix sum for new row pointers
  thrust::exclusive_scan(flag_ptr, flag_ptr+A.nnz, flag_ptr);

  // Set new row pointers
  // row2[i] = flag[ row[i] ], for n elements
  thrust::copy(
    thrust::make_permutation_iterator(flag_ptr, row_ptr),
    thrust::make_permutation_iterator(flag_ptr, row_ptr+A.HDIM),
    row2_ptr);

  // Release flag space
  thrust::device_free(flag_ptr);

}
