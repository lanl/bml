#include "bml_introspection_ellpack.h"

#include <stdlib.h>

/** Return the matrix size.
 *
 * \param A The matrix.
 * \return The matrix size.
 */
int
bml_get_size_ellpack (const bml_matrix_ellpack_t * A)
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
