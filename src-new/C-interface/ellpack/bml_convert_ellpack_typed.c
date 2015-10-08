#include "../typed.h"
#include "bml_allocate.h"
#include "bml_allocate_ellpack.h"
#include "bml_convert.h"
#include "bml_convert_ellpack.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group
 *
 * \param N The number of rows/columns
 * \param matrix_precision The real precision
 * \param A The dense matrix
 * \return The bml matrix
 */
bml_matrix_ellpack_t *
TYPED_FUNC(bml_convert_from_dense_ellpack)(
    const int N,
    const void *A,
    const double threshold,
    const int M)
{
    bml_matrix_ellpack_t *A_bml =
        TYPED_FUNC(bml_zero_matrix_ellpack)(N, M);

    int *A_index = A_bml->index;
    int *nnz = A_bml->nnz;

    REAL_T *dense_A = (REAL_T*) A;
    REAL_T *A_value = A_bml->value;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (fabs(dense_A[j + i * N]) > (REAL_T) threshold)
            {
                A_value[nnz[j] + i * M] = dense_A[j + i * N];
                A_index[nnz[j] + i * M] = i;
                nnz[j]++;
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
void *
TYPED_FUNC(bml_convert_to_dense_ellpack)(
    const bml_matrix_ellpack_t * A)
{
    int *A_index = A->index;
    int *nnz = A->nnz;

    int N = A->N;
    int M = A->M;

    REAL_T *A_dense = bml_allocate_memory(sizeof(REAL_T) * N * N);
    REAL_T *A_value = A->value;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < nnz[i]; j++)
        {
            A_dense[A_index[j + i * M] + i * N] =
                A_value[j + i * M];
        }
    }
    return A_dense;
}
