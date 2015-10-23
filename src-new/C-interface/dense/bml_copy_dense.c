#include "bml_allocate.h"
#include "bml_allocate_dense.h"
#include "bml_copy.h"
#include "bml_copy_dense.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/** Copy a dense matrix - result in new matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_dense_t *
bml_copy_dense_new(
    const bml_matrix_dense_t * A)
{
    bml_matrix_dense_t *B = NULL;

    assert(A != NULL);

    switch (A->matrix_precision)
    {
        case single_real:
            B = bml_copy_dense_new_single_real(A);
            break;
        case double_real:
            B = bml_copy_dense_new_double_real(A);
            break;
        case single_complex:
            B = bml_copy_dense_new_single_complex(A);
            break;
        case double_complex:
            B = bml_copy_dense_new_double_complex(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return B;
}

/** Copy a dense matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \param B Copy of matrix A
 */
void
bml_copy_dense(
    const bml_matrix_dense_t * A,
    bml_matrix_dense_t * B)
{
    assert(A != NULL);

    switch (A->matrix_precision)
    {
        case single_real:
            bml_copy_dense_single_real(A, B);
            break;
        case double_real:
            bml_copy_dense_double_real(A, B);
            break;
        case single_complex:
            bml_copy_dense_single_complex(A, B);
            break;
        case double_complex:
            bml_copy_dense_double_complex(A, B);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
