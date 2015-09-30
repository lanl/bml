#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "dense/bml_introspection_dense.h"
#include "ellpack/bml_introspection_ellpack.h"

#include <stdlib.h>

/** Returns the matrix type.
 *
 * If the matrix is not initialized yet, a type of "unitialized" is returned.
 *
 * \param A The matrix.
 * \return The matrix type.
 */
bml_matrix_type_t
bml_get_type(
    const bml_matrix_t * A)
{
    const bml_matrix_type_t *matrix_type = A;
    if (A != NULL)
    {
        return *matrix_type;
    }
    else
    {
        return uninitialized;
    }
}

/** Return the matrix size.
 *
 * \param A The matrix.
 * \return The matrix size.
 */
int
bml_get_size(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case uninitialized:
        case dense:
            return bml_get_size_dense(A);
            break;
        case ellpack:
            return bml_get_size_ellpack(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return -1;
}
