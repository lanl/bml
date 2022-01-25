#ifdef BML_USE_MAGMA
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
    double alpha = 1.0;

    REAL_T sum = 0.0;
#ifdef BML_USE_MAGMA            //do work on CPU for now...
    MAGMA_T *A_matrix = bml_allocate_memory(sizeof(MAGMA_T) * A->N * A->N);
    MAGMA(getmatrix) (A->N, A->N, A->matrix, A->ld, A_matrix, A->N,
                      bml_queue());
    MAGMA_T *B_matrix = bml_allocate_memory(sizeof(MAGMA_T) * B->N * B->N);
    MAGMA(getmatrix) (B->N, B->N, B->matrix, B->ld, B_matrix, B->N,
                      bml_queue());
    MAGMA_T *C_matrix = bml_allocate_memory(sizeof(MAGMA_T) * C->N * C->N);
    MAGMA(getmatrix) (C->N, C->N, C->matrix, C->ld, C_matrix, C->N,
                      bml_queue());

#else
    REAL_T *A_matrix = A->matrix;
    REAL_T *B_matrix = B->matrix;
    REAL_T *C_matrix = C->matrix;
#endif

    int *A_localRowMin = A->domain->localRowMin;
    int *A_localRowMax = A->domain->localRowMax;

#ifdef BML_USE_MAGMA
    MAGMA_T alpha_ = MAGMACOMPLEX(MAKE) (alpha, 0.);
#else
    REAL_T alpha_ = (REAL_T) alpha;
#endif

    int myRank = bml_getMyRank();

#pragma omp parallel for                        \
  shared(alpha_)                         \
  shared(N, A_matrix, B_matrix)                 \
  shared(A_localRowMin, A_localRowMax, myRank)  \
                                //for (int i = 0; i < N * N; i++)
    for (int i = A_localRowMin[myRank] * N; i < A_localRowMax[myRank] * N;
         i++)
    {
#ifdef BML_USE_MAGMA
        MAGMA_T temp =
            MAGMACOMPLEX(MUL) (MAGMACOMPLEX(MUL) (alpha_, A_matrix[i]),
                               B_matrix[i]);
#else
        REAL_T temp = alpha_ * A_matrix[i] * B_matrix[i];
#endif
        C_matrix[i] = temp;     //* temp;
    }
#ifdef BML_USE_MAGMA
    free(A_matrix);
    free(B_matrix);
    free(C_matrix);
#endif
}
