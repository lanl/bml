#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_normalize.h"
#include "dense/bml_normalize_dense.h"
#include "ellpack/bml_normalize_ellpack.h"

#include <stdlib.h>

/** Normalize matrix given Gershgorin bounds.
 *
 ** \ingroup normalize_group_C
 *
 * \param A Matrix to scale
 * \param maxeval Calculated max value
 * \param maxminusmin Calculated max-min value
 */
void
bml_normalize(
    bml_matrix_t * A,
    const double maxeval,
    const double maxminusmin)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_normalize_dense(A, maxeval, maxminusmin);
            break;
        case ellpack:
            return bml_normalize_ellpack(A, maxeval, maxminusmin);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Calculate Gershgorin bounds.
 *
 * \ingroup normalize_group_C
 *
 * \param A Matrix to scale
 * returns maxeval Calculated max value
 * returns maxminusmin Calculated max-min value
 */
void *
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
