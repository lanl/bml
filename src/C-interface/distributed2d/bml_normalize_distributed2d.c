#include "../bml_logger.h"
#include "../bml_normalize.h"
#include "../bml_types.h"
#include "../bml_introspection.h"
#include "bml_normalize_distributed2d.h"
#include "bml_types_distributed2d.h"

#include <stdlib.h>
#include <string.h>

/** Normalize distributed2d matrix given gershgorin bounds.
 *
 *  \ingroup normalize_group
 *
 *  \param A The matrix
 *  \param mineval Calculated min value
 *  \param maxeval Calculated max value
 */
void
bml_normalize_distributed2d(
    bml_matrix_distributed2d_t * A,
    double mineval,
    double maxeval)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_normalize_distributed2d_single_real(A, mineval, maxeval);
            break;
        case double_real:
            bml_normalize_distributed2d_double_real(A, mineval, maxeval);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_normalize_distributed2d_single_complex(A, mineval, maxeval);
            break;
        case double_complex:
            bml_normalize_distributed2d_double_complex(A, mineval, maxeval);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

/** Calculate gershgorin bounds for an distributed2d matrix.
 *
 *  \ingroup normalize_group
 *
 *  \param A The matrix
 *  returns mineval Calculated min value
 *  returns maxeval Calculated max value
 */
void *
bml_gershgorin_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            return bml_gershgorin_distributed2d_single_real(A);
            break;
        case double_real:
            return bml_gershgorin_distributed2d_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_gershgorin_distributed2d_single_complex(A);
            break;
        case double_complex:
            return bml_gershgorin_distributed2d_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
