#include "../../macros.h"
#include "../typed.h"
#include "bml_allocate.h"
#include "bml_allocate_dense.h"
#include "bml_convert_dense.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <complex.h>
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
bml_matrix_dense_t *TYPED_FUNC(
    bml_import_from_dense_dense) (
    const bml_dense_order_t order,
    const int N,
    const void *A,
    const bml_distribution_mode_t distrib_mode)
{
    bml_matrix_dense_t *A_bml =
        TYPED_FUNC(bml_zero_matrix_dense) (N, distrib_mode);

    switch (order)
    {
        case dense_row_major:
            memcpy(A_bml->matrix, A, sizeof(REAL_T) * N * N);
            break;
        case dense_column_major:
        {
            REAL_T *A_ptr = (REAL_T *) A;
            REAL_T *B_ptr = (REAL_T *) A_bml->matrix;
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    B_ptr[ROWMAJOR(i, j, N, N)] = A_ptr[COLMAJOR(i, j, N, N)];
                }
            }
            break;
        }
        default:
            LOG_ERROR("logic error\n");
            break;
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
    bml_export_to_dense_dense) (
    const bml_matrix_dense_t * A,
    const bml_dense_order_t order)
{
    REAL_T *A_dense = bml_allocate_memory(sizeof(REAL_T) * A->N * A->N);
    switch (order)
    {
        case dense_row_major:
            memcpy(A_dense, A->matrix, sizeof(REAL_T) * A->N * A->N);
            break;
        case dense_column_major:
        {
            REAL_T *B_ptr = (REAL_T *) A->matrix;
            for (int i = 0; i < A->N; i++)
            {
                for (int j = 0; j < A->N; j++)
                {
                    A_dense[COLMAJOR(i, j, A->N, A->N)] =
                        B_ptr[ROWMAJOR(i, j, A->N, A->N)];
                }
            }
            break;
        }
        default:
            LOG_ERROR("logic error\n");
            break;
    }
    return A_dense;
}
