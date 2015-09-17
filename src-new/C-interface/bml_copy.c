#include "bml_copy.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_copy_dense.h"
#include "ellpack/bml_copy_ellpack.h"

#include <stdlib.h>

/** Copy a matrix.
 *
 * \ingroup copy_group_C
 *
 * \param A Matrix to copy
 * \return  A Copy of A
 */
bml_matrix_t *bml_copy(const bml_matrix_t *A)
{
    bml_matrix_t *B = NULL;

    switch(bml_get_type(A)) {
    case dense:
        B = bml_copy_dense(A);
        break;
    case ellpack:
        B = bml_copy_ellpack(A);
        break;
    default:
        LOG_ERROR("unknown matrix type\n");
        break;
    }
    return B;
}
