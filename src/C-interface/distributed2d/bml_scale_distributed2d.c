#include "../bml_logger.h"
#include "../bml_scale.h"
#include "../bml_types.h"
#include "../bml_introspection.h"
#include "bml_scale_distributed2d.h"
#include "bml_types_distributed2d.h"
#include "bml_allocate_distributed2d.h"

/** Scale an distributed2d matrix - result is a new matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \return B, scaled version of matrix A.
 */
bml_matrix_distributed2d_t *
bml_scale_distributed2d_new(
    void *scale_factor,
    bml_matrix_distributed2d_t * A)
{
    bml_matrix_distributed2d_t *B =
        bml_zero_matrix_distributed2d(bml_get_type(A->matrix),
                                      bml_get_precision(A->matrix), A->N,
                                      A->M);

    bml_scale(scale_factor, A->matrix, B->matrix);

    return B;
}

/** Scale an distributed2d matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \param B Scaled version of matrix A
 */
void
bml_scale_distributed2d(
    void *scale_factor,
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B)
{
    bml_scale(scale_factor, A->matrix, B->matrix);
}

void
bml_scale_inplace_distributed2d(
    void *scale_factor,
    bml_matrix_distributed2d_t * A)
{
    bml_scale_inplace(scale_factor, A->matrix);
}
