#include "bml_allocate.h"
#include "bml_allocate_dense.h"
#include "bml_logger.h"
#include "bml_gershgorin.h"
#include "bml_gershgorin_dense.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/** Calculate Gershgorin bounds for a dense matrix.
 *
 *  \ingroup gershgorin_group
 *
 *  \param A The matrix
 *  \param maxeval Calculated max value
 *  \param maxminusmin Calculated max-min value
 *  \param threshold Threshold for matrix
 */
void
bml_gershgorin_dense(
    const bml_matrix_dense_t * A,
    double maxeval,
    double minminusmax,
    const double threshold)
{
    assert(A != NULL);

    switch (A->matrix_precision)
    {
        case single_real:
            bml_gershgorin_dense_single_real(A, maxeval, maxminusmin, threshold);
            break;
        case double_real:
            bml_gershgorin_dense_double_real(A, maxeval, maxminusmin, threshold);
            break;
        case single_complex:
            bml_gershgorin_dense_single_complex(A, maxeval, maxminusmin, threshold);
            break;
        case double_complex:
            bml_gershgorin_dense_double_complex(A, maxeval, maxminusmin, threshold);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
