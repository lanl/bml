#include "../typed.h"
#include "bml_allocate.h"
#include "bml_threshold.h"
#include "bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_threshold_dense.h"
#include "bml_types_dense.h"

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
    const bml_matrix_dense_t * A,
    const double threshold)
{
    int N = A->N;
    bml_matrix_dense_t *B = TYPED_FUNC(bml_zero_matrix_dense) (N);
    REAL_T *A_matrix = A->matrix;
    REAL_T *B_matrix = B->matrix;

#pragma omp parallel for default(none) shared(N, A_matrix, B_matrix)
    for (int i = 0; i < N * N; i++)
    {
        if (fabs(A_matrix[i]) > threshold)
        {
            B_matrix[i] = A_matrix[i];
        }
    }
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
    const bml_matrix_dense_t * A,
    const double threshold)
{
    int N = A->N;
    REAL_T *A_matrix = A->matrix;

#pragma omp parallel for default(none) shared(N, A_matrix)
    for (int i = 0; i < N * N; i++)
    {
        if (fabs(A_matrix[i]) < threshold)
        {
            A_matrix[i] = (REAL_T) 0.0;
        }
    }
}
