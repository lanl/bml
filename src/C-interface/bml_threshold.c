#include "bml_threshold.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_threshold_dense.h"
#include "ellpack/bml_threshold_ellpack.h"

#include <stdlib.h>

/** Threshold matrix.
 *
 * \ingroup threshold_group_C
 *
 * \param A Matrix to be thresholded
 * \param threshold Threshold value
 * \return  Thresholded A
 */
bml_matrix_t *
bml_threshold_new(
    const bml_matrix_t * A,
    const double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_threshold_new_dense(A, threshold);
            break;
        case ellpack:
            return bml_threshold_new_ellpack(A, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return NULL;
}

/** Threshold matrix.
 *
 * \ingroup threshold_group_C
 *
 * \param A Matrix to be thresholded
 * \param threshold Threshold value
 * \return  Thresholded A
 */
void
bml_threshold(
    const bml_matrix_t * A,
    const double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_threshold_dense(A, threshold);
            break;
        case ellpack:
            bml_threshold_ellpack(A, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
