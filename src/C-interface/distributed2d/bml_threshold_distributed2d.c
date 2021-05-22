#include "../bml_threshold.h"
#include "../bml_types.h"
#include "../bml_introspection.h"
#include "../bml_copy.h"
#include "bml_threshold_distributed2d.h"
#include "bml_types_distributed2d.h"
#include "bml_allocate_distributed2d.h"

#include <assert.h>

/** Threshold a matrix.
 *
 *  \ingroup threshold_group
 *
 *  \param A The matrix to be thresholded
 *  \param threshold Threshold value
 *  \return the thresholded A
 */
bml_matrix_distributed2d_t
    * bml_threshold_new_distributed2d(bml_matrix_distributed2d_t * A,
                                      double threshold)
{
    assert(A->M > 0);

    bml_matrix_distributed2d_t *B =
        bml_zero_matrix_distributed2d(bml_get_type(A->matrix),
                                      bml_get_precision(A->matrix), A->N,
                                      A->M);
    // copy local block
    bml_copy(A->matrix, B->matrix);

    bml_threshold(B->matrix, threshold);

    return B;
}
