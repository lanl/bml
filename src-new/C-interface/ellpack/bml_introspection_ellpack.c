#include "bml_introspection_ellpack.h"

#include <stdlib.h>

/** Return the matrix precision.
 *
 * \param A The matrix.
 * \return The matrix precision.
 */
bml_matrix_precision_t bml_get_precision_ellpack(
    const bml_matrix_ellpack_t * A)
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
bml_get_N_ellpack(
    const bml_matrix_ellpack_t * A)
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
int bml_get_M_ellpack(
    const bml_matrix_ellpack_t * A)
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
