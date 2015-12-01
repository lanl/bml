#include "bml_logger.h"
#include "bml_gershgorin.h"
#include "bml_gershgorin_ellpack.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>

/** Calculate gershgorin bounds for an ellpack matrix.
 *
 *  \ingroup gershgorin_group
 *
 *  \param A The matrix
 *  \param maxeval Calculated max value
 *  \param maxminusmin Calculated max-min value
 *  \param threshold Threshold for matrix
 */
void*
bml_gershgorin_ellpack(
    const bml_matrix_ellpack_t * A)
{
    switch (A->matrix_precision)
    {
        case single_real:
            return bml_gershgorin_ellpack_single_real(A);
            break;
        case double_real:
            return bml_gershgorin_ellpack_double_real(A);
            break;
        case single_complex:
            return bml_gershgorin_ellpack_single_complex(A);
            break;
        case double_complex:
            return bml_gershgorin_ellpack_double_complex(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
