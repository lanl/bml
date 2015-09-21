#include "bml_copy.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_copy_dense.h"
#include "ellpack/bml_copy_ellpack.h"

#include <stdlib.h>

/** Copy a matrix - result is a new matrix.
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

/** Copy a matrix - result is an existing matrix.
 *
 * \ingroup copy_group_C
 *
 * \param A Matrix to copy
 * \param B Copy of Matrix A
 */
void bml_copy(const bml_matrix_t *A, const bml_matrix_t *B)
{

    switch(bml_get_type(A)) {
    case dense:
        bml_copy_dense(A, B);
        break;
    case ellpack:
        bml_copy_ellpack(A, B);
        break;
    default:
        LOG_ERROR("unknown matrix type\n");
        break;
    }
}
