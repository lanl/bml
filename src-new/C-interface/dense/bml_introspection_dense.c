#include "bml_introspection_dense.h"

#include <stdlib.h>

/** Return the matrix size.
 *
 * \param A The matrix.
 * \return The matrix size.
 */
int
bml_get_size_dense(
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
