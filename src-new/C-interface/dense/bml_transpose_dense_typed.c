#include "../typed.h"
#include "bml_allocate.h"
#include "bml_transpose.h"
#include "bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_transpose_dense.h"
#include "bml_types_dense.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Transpose a matrix.
 *
 *  \ingroup transpose_group
 *
 *  \param A The matrix to be transposed
 *  \return The transposed A
 */
bml_matrix_dense_t *TYPED_FUNC(
    bml_transpose_new_dense) (
    const bml_matrix_dense_t * A)
{
    bml_matrix_dense_t *B = TYPED_FUNC(bml_zero_matrix_dense) (A->N);
    REAL_T *A_matrix = A->matrix;
    REAL_T *B_matrix = B->matrix;

#pragma omp parallel for default(none) shared(A->N, A_matrix, B_matrix)
    for (int i = 0; i < A->N; i++)
    {
        for (int j = 0; j < A->N; j++)
        {
            B_matrix[i * A->N + j] = A_matrix[j * A->N + i];
        }
    }
    return B;
}

/** Transpose a matrix in place.
 *
 *  \ingroup transpose_group
 *
 *  \param A The matrix to be transposed
 *  \return The transposed A
 */
void TYPED_FUNC(
    bml_transpose_dense) (
    const bml_matrix_dense_t * A)
{
    REAL_T *A_matrix = A->matrix;
    REAL_T tmp;

#pragma omp parallel for default(none) private(tmp) shared(A->N, A_matrix)
    for (int i = 0; i < A->N - 1; i++)
    {
        for (int j = i + 1; j < A->N; j++)
        {
            if (i != j)
            {
                tmp = A_matrix[i * A->N + j];
                A_matrix[i * A->N + j] = A_matrix[j * A->N + i];
                A_matrix[j * A->N + i] = tmp;
            }
        }
    }
}
