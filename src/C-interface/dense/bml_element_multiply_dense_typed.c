#ifdef BML_USE_MAGMA
//define boolean data type needed by magma
#include <stdbool.h>
#include "magma_v2.h"
#endif

#include "../../internal-blas/bml_gemm.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_element_multiply.h"
#include "../bml_parallel.h"
#include "../bml_trace.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_element_multiply_dense.h"
#include "bml_trace_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MKL_GPU
#include "stdio.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#endif

/** Element-wise Matrix multiply (Hadamard product)
 *
 * \f$ C_{ij} \leftarrow A_{ij} * B_{ij} \f$
 *
 * \ingroup norm_group
 *
 * \param A The matrix A
 * \param B The matrix B
 * \param C Matrix C
 */
void TYPED_FUNC(
    bml_element_multiply_AB_dense) (
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C)
{
    int N = A->N;

#ifdef BML_USE_MAGMA            //do work on CPU for now...
    REAL_T *A_matrix = bml_allocate_memory(sizeof(REAL_T) * A->N * A->N);
    MAGMA(getmatrix) (A->N, A->N, A->matrix, A->ld, (MAGMA_T *) A_matrix,
                      A->N, bml_queue());
    REAL_T *B_matrix = bml_allocate_memory(sizeof(REAL_T) * B->N * B->N);
    MAGMA(getmatrix) (B->N, B->N, B->matrix, B->ld, (MAGMA_T *) B_matrix,
                      B->N, bml_queue());
    REAL_T *C_matrix = bml_allocate_memory(sizeof(REAL_T) * C->N * C->N);
    MAGMA(getmatrix) (C->N, C->N, C->matrix, C->ld, (MAGMA_T *) C_matrix,
                      C->N, bml_queue());

#else
    REAL_T *A_matrix = A->matrix;
    REAL_T *B_matrix = B->matrix;
    REAL_T *C_matrix = C->matrix;

#ifdef MKL_GPU
// pull from GPU
#pragma omp target update from(A_matrix[0:N*N], B_matrix[0:N*N])
#endif

#endif

    int *A_localRowMin = A->domain->localRowMin;
    int *A_localRowMax = A->domain->localRowMax;

    int myRank = bml_getMyRank();

#pragma omp parallel for                        \
  shared(N, C_matrix, A_matrix, B_matrix)                 \
  shared(A_localRowMin, A_localRowMax, myRank)  \
                                //for (int i = 0; i < N * N; i++)
    for (int i = A_localRowMin[myRank] * N; i < A_localRowMax[myRank] * N;
         i++)
    {
        C_matrix[i] = A_matrix[i] * B_matrix[i];
    }
#ifdef BML_USE_MAGMA
    MAGMA(setmatrix) (C->N, C->N, (MAGMA_T *) C_matrix, C->N, C->matrix,
                      C->ld, bml_queue());

    bml_free_memory(A_matrix);
    bml_free_memory(B_matrix);
    bml_free_memory(C_matrix);
#endif
#ifdef MKL_GPU
// push back to GPU
#pragma omp target update to(C_matrix[0:N*N])
#endif
}
