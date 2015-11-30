#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_gershgorin.h"
#include "dense/bml_gershgorin_dense.h"
#include "ellpack/bml_gershgorin_ellpack.h"

#include <stdlib.h>

/** Calculate Gershgorin bounds.
 *
 * \ingroup gershgorin_group_C
 *
 * \param scale_factor Scale factor for A
 * \param A Matrix to scale
 * \param maxeval Calculated max value
 * \param maxminusmin Calculated max-min value
 * \param threshold Threshold for A
 */
void*
bml_gershgorin(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_gershgorin_dense(A);
            break;
        case ellpack:
            return bml_gershgorin_ellpack(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return NULL;
}

