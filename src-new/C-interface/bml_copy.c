#include "bml_allocate.h"
#include "bml_copy.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_copy_dense.h"
#include "ellpack/bml_copy_ellpack.h"

#include <assert.h>
#include <stdlib.h>

/** Copy a matrix - result is a new matrix.
 *
 * \ingroup copy_group_C
 *
 * \param A Matrix to copy
 * \return  A Copy of A
 */
bml_matrix_t *
bml_copy_new(
    const bml_matrix_t * A)
{
    bml_matrix_t *B = NULL;

    switch (bml_get_type(A))
    {
        case dense:
            B = bml_copy_dense_new(A);
            break;
        case ellpack:
            B = bml_copy_ellpack_new(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return B;
}

/** Copy a matrix.
 *
 * \param A Matrix to copy
 * \param B Copy of Matrix A
 */
void
bml_copy(
    const bml_matrix_t * A,
    bml_matrix_t ** B)
{
    assert(A != NULL);
    LOG_DEBUG("copying matrix\n");
    *B = bml_zero_matrix(bml_get_type(A), bml_get_precision(A), bml_get_N(A),
                         bml_get_M(A));
    switch (bml_get_type(A))
    {
        case dense:
            bml_copy_dense(A, *B);
            break;
        case ellpack:
            bml_copy_ellpack(A, *B);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
