#include "bml_introspection_ellsort.h"

#include <stdlib.h>

/** Return the matrix precision.
 *
 * \param A The matrix.
 * \return The matrix precision.
 */
bml_matrix_precision_t
bml_get_precision_ellsort(
    const bml_matrix_ellsort_t * A)
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

/** Return the matrix distribution mode.
 *
 * \param A The matrix.
 * \return The matrix distribution mode.
 */
bml_distribution_mode_t
bml_get_distribution_mode_ellsort(
    const bml_matrix_ellsort_t * A)
{
    if (A != NULL)
    {
        return A->distribution_mode;
    }
    else
    {
        return -1;
    }
}

/** Return the matrix size.
 *
 * \param A The matrix.
 * \return The matrix size.
 */
int
bml_get_N_ellsort(
    const bml_matrix_ellsort_t * A)
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
bml_get_M_ellsort(
    const bml_matrix_ellsort_t * A)
{
    if (A != NULL)
    {
        return A->M;
    }
    else
    {
        return -1;
    }
}

/** Return the bandwidth of a row in the matrix.
 *
 * \param A The bml matrix.
 * \param i The row index.
 * \return The bandwidth of row i.
 */
int
bml_get_row_bandwidth_ellsort(
    const bml_matrix_ellsort_t * A,
    const int i)
{
    return A->nnz[i];
}

/** Return the bandwidth of a matrix.
 *
 * \param A The bml matrix.
 * \return The bandwidth of row i.
 */
int
bml_get_bandwidth_ellsort(
    const bml_matrix_ellsort_t * A)
{
    int max_bandwidth = 0;
    for (int i = 0; i < A->N; i++)
    {
        max_bandwidth =
            (A->nnz[i] > max_bandwidth ? A->nnz[i] : max_bandwidth);
    }
    return max_bandwidth;
}
