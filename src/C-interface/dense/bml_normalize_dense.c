#include "bml_allocate.h"
#include "bml_allocate_dense.h"
#include "bml_logger.h"
#include "bml_normalize.h"
#include "bml_normalize_dense.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/** Normalize dense matrix given Gershgorin bounds.
 *
 *  \ingroup normalize_group
 *
 *  \param A The matrix
 *  \param maxeval Calculated max value
 *  \param maxminusmin Calculated max-min value
 */
void
bml_normalize_dense(
    bml_matrix_dense_t * A,
    const double maxeval,
    const double maxminusmin)
{
    assert(A != NULL);

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_normalize_dense_single_real(A, maxeval, maxminusmin);
            break;
        case double_real:
            return bml_normalize_dense_double_real(A, maxeval, maxminusmin);
            break;
        case single_complex:
            return bml_normalize_dense_single_complex(A, maxeval, maxminusmin);
            break;
        case double_complex:
            return bml_normalize_dense_double_complex(A, maxeval, maxminusmin);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

/** Calculate Gershgorin bounds for a dense matrix.
 *
 *  \ingroup normalize_group
 *
 *  \param A The matrix
 *  returns maxeval Calculated max value
 *  returns maxminusmin Calculated max-min value
 */
void *
bml_gershgorin_dense(
    const bml_matrix_dense_t * A)
{
    assert(A != NULL);

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_gershgorin_dense_single_real(A);
            break;
        case double_real:
            return bml_gershgorin_dense_double_real(A);
            break;
        case single_complex:
            return bml_gershgorin_dense_single_complex(A);
            break;
        case double_complex:
            return bml_gershgorin_dense_double_complex(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
