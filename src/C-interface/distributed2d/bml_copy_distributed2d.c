#include "../bml_copy.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "../bml_introspection.h"
#include "bml_copy_distributed2d.h"
#include "bml_types_distributed2d.h"
#include "bml_allocate_distributed2d.h"

#include <assert.h>

/** Copy an distributed2d matrix - result is a new matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_distributed2d_t *
bml_copy_distributed2d_new(
    bml_matrix_distributed2d_t * A)
{
    assert(A->M > 0);

    bml_matrix_distributed2d_t *B =
        bml_zero_matrix_distributed2d(bml_get_type(A->matrix),
                                      bml_get_precision(A->matrix), A->N,
                                      A->M);

    // copy local block
    bml_copy(A->matrix, B->matrix);

    assert(B != NULL);
    return B;
}

/** Copy an distributed2d matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \param B Copy of matrix A
 */
void
bml_copy_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B)
{
    bml_copy(A->matrix, B->matrix);
}
