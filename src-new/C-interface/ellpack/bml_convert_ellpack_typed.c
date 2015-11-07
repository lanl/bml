#include "../macros.h"
#include "../typed.h"
#include "bml_allocate.h"
#include "bml_allocate_ellpack.h"
#include "bml_convert.h"
#include "bml_convert_ellpack.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group
 *
 * \param N The number of rows/columns
 * \param matrix_precision The real precision
 * \param A The dense matrix
 * \return The bml matrix
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_convert_from_dense_ellpack) (
    const int N,
    const void *A,
    const double threshold,
    const int M)
{
    bml_matrix_ellpack_t *A_bml = TYPED_FUNC(bml_zero_matrix_ellpack) (N, M);

    int *A_index = A_bml->index;
    int *A_nnz = A_bml->nnz;

    REAL_T *dense_A = (REAL_T *) A;
    REAL_T *A_value = A_bml->value;

#pragma omp parallel for default(none) shared(A_value,A_index,A_nnz,dense_A)
    for (int i = 0; i < N; i++)
    {
        A_nnz[i] = 1;
        for (int j = 0; j < N; j++)
        {
            if (is_above_threshold(dense_A[ROWMAJOR(i, j, N)], threshold))
            {
                if (i == j)
                {
                    A_value[ROWMAJOR(i, 0, M)] = dense_A[ROWMAJOR(i, j, N)];
                    A_index[ROWMAJOR(i, 0, M)] = j;
                }
                else
                {
                    A_value[ROWMAJOR(i, A_nnz[i], M)] =
                        dense_A[ROWMAJOR(i, j, N)];
                    A_index[ROWMAJOR(i, A_nnz[i], M)] = j;
                    A_nnz[i]++;
                }
            }
        }
    }

    return A_bml;
}

/** Convert a bml matrix into a dense matrix.
 *
 * \ingroup convert_group
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
void *TYPED_FUNC(
    bml_convert_to_dense_ellpack) (
    const bml_matrix_ellpack_t * A)
{
    int *A_index = A->index;
    int *A_nnz = A->nnz;

    int N = A->N;
    int M = A->M;

    REAL_T *A_dense = bml_allocate_memory(sizeof(REAL_T) * N * N);
    REAL_T *A_value = A->value;

#pragma omp parallel for default(none) shared(N,M,A_value,A_index,A_nnz,A_dense)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < A_nnz[i]; j++)
        {
            A_dense[ROWMAJOR(i, A_index[ROWMAJOR(i, j, M)], N)] =
                A_value[ROWMAJOR(i, j, M)];
        }
    }
    return A_dense;
}
