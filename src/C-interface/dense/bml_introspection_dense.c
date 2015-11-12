#include "bml_introspection_dense.h"

#include <stdlib.h>

/** Return the matrix precision.
 *
 * \param A The matrix.
 * \return The matrix precision.
 */
bml_matrix_precision_t
bml_get_precision_dense(
    const bml_matrix_dense_t * A)
{
    if (A != NULL)
    {
        return A->matrix_precision;
    }
    else
    {
        return precision_uninitialized;
    }
}

/** Return the matrix size.
 *
 * \param A The matrix.
 * \return The matrix size.
 */
int
bml_get_N_dense(
    const bml_matrix_dense_t * A)
{
    if (A != NULL)
    {
        return A->N;
    }
    else
    {
        return -1;
    }
}

/** Return the matrix parameter M.
 *
 * \param A The matrix.
 * \return The matrix parameter M.
 */
int
bml_get_M_dense(
    const bml_matrix_dense_t * A)
{
    return 0;
}

/** Return the bandwidth of a row in the matrix.
 *
 * \param A The bml matrix.
 * \param i The row index.
 * \return The bandwidth of row i.
 */
int
bml_get_row_bandwidth_dense(
    const bml_matrix_dense_t * A,
    const int i)
{
    switch (bml_get_precision_dense(A))
    {
        case single_real:
            return bml_get_row_bandwidth_dense_single_real(A, i);
            break;
        case double_real:
            return bml_get_row_bandwidth_dense_double_real(A, i);
            break;
        case single_complex:
            return bml_get_row_bandwidth_dense_single_complex(A, i);
            break;
        case double_complex:
            return bml_get_row_bandwidth_dense_double_complex(A, i);
            break;
    }
}
