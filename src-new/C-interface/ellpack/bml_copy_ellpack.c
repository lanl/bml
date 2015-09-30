#include "bml_copy.h"
#include "bml_copy_ellpack.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>

/** Copy an ellpack matrix - result is a new matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_ellpack_t *
bml_copy_ellpack_new(
    const bml_matrix_ellpack_t * A)
{
    bml_matrix_ellpack_t *B = NULL;

    switch (B->matrix_precision)
    {
        case single_real:
            B = bml_copy_ellpack_new_single_real(A);
            break;
        case double_real:
            B = bml_copy_ellpack_new_double_real(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return B;
}

/** Copy an ellpack matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \param B Copy of matrix A
 */
void
bml_copy_ellpack(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B)
{

    switch (A->matrix_precision)
    {
        case single_real:
            bml_copy_ellpack_single_real(A, B);
            break;
        case double_real:
            bml_copy_ellpack_double_real(A, B);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
