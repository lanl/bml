#include "bml_transpose.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_transpose_dense.h"
#include "ellpack/bml_transpose_ellpack.h"

#include <stdlib.h>

/** Transpose matrix.
 *
 * \ingroup transpose_group_C
 *
 * \param A Matrix to be transposed
 * \return  Transposed A
 */
bml_matrix_t *
bml_transpose_new(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_transpose_new_dense(A);
            break;
        case ellpack:
            return bml_transpose_new_ellpack(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return NULL;
}

/** Transpose matrix.
 *
 * \ingroup transpose_group_C
 *
 * \param A Matrix to be transposed
 * \return  Transposed A
 */
void
bml_transpose(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_transpose_dense(A);
            break;
        case ellpack:
            bml_transpose_ellpack(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
