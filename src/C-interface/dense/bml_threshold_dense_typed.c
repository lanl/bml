#ifdef BML_USE_MAGMA
#include "magma_v2.h"
#include "../bml_export.h"
#include "bml_export_dense.h"
#endif

#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_threshold.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_threshold_dense.h"
#include "bml_types_dense.h"
#include "../bml_logger.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Threshold a matrix.
 *
 *  \ingroup threshold_group
 *
 *  \param A The matrix to be thresholded
 *  \param threshold Threshold value
 *  \return The thresholded A
 */
bml_matrix_dense_t *TYPED_FUNC(
    bml_threshold_new_dense) (
    bml_matrix_dense_t * A,
    double threshold)
{
#ifdef BML_USE_MAGMA
    LOG_ERROR
        ("bml_threshold_new_dense() not implemented for MAGMA matrices\n");
#endif

    int N = A->N;
    bml_matrix_dimension_t matrix_dimension = { A->N, A->N, A->N };
    bml_matrix_dense_t *B =
        TYPED_FUNC(bml_zero_matrix_dense) (matrix_dimension,
                                           A->distribution_mode);
    REAL_T *A_matrix = A->matrix;
    REAL_T *B_matrix = B->matrix;

    int *A_localRowMin = A->domain->localRowMin;
    int *A_localRowMax = A->domain->localRowMax;

    int myRank = bml_getMyRank();

#ifdef MKL_GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif
#pragma omp parallel for                        \
  shared(N, A_matrix, B_matrix)                 \
  shared(A_localRowMin, A_localRowMax, myRank)
    //for (int i = 0; i < N * N; i++)
    for (int i = A_localRowMin[myRank] * N; i < A_localRowMax[myRank] * N;
         i++)
    {
        if (is_above_threshold(A_matrix[i], (REAL_T) threshold))
        {
            B_matrix[i] = A_matrix[i];
        }
    }
#ifdef MKL_GPU
#pragma omp target update to(B_matrix[0:N*N])
#endif
    return B;
}

/** Threshold a matrix in place.
 *
 *  \ingroup threshold_group
 *
 *  \param A The matrix to be thresholded
 *  \param threshold Threshold value
 *  \return The thresholded A
 */
void TYPED_FUNC(
    bml_threshold_dense) (
    bml_matrix_dense_t * A_bml,
    double threshold)
{
    int N = A_bml->N;
#ifdef BML_USE_MAGMA
    REAL_T *A_matrix = bml_export_to_dense(A_bml, dense_row_major);
#else
    REAL_T *A_matrix = A_bml->matrix;
#endif

    int *A_localRowMin = A_bml->domain->localRowMin;
    int *A_localRowMax = A_bml->domain->localRowMax;

    int myRank = bml_getMyRank();

#ifdef MKL_GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif
#pragma omp parallel for                        \
  shared(N, A_matrix)                           \
  shared(A_localRowMin, A_localRowMax, myRank)
    //for (int i = 0; i < N * N; i++)
    for (int i = A_localRowMin[myRank] * N; i < A_localRowMax[myRank] * N;
         i++)
    {
        if (!is_above_threshold(A_matrix[i], (REAL_T) threshold))
        {
            A_matrix[i] = (REAL_T) 0.0;
        }
    }
#ifdef MKL_GPU
#pragma omp target update to(A_matrix[0:N*N])
#endif
#ifdef BML_USE_MAGMA
    MAGMA(setmatrix) (N, N, (MAGMA_T *) A_matrix, N, A_bml->matrix, A_bml->ld,
                      bml_queue());
#endif
}
