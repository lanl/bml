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
 * \param mineval Calculated min value
 * \param maxeval Calculated max value
 */
void
bml_normalize(
    bml_matrix_t * A,
    const double mineval,
    const double maxeval)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_normalize_dense(A, mineval, maxeval);
            break;
        case ellpack:
            bml_normalize_ellpack(A, mineval, maxeval);
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
 * returns mineval Calculated min value
 * returns maxeval Calculated max value
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

/** Calculate Gershgorin bounds for partial matrix.
 *
 * \ingroup normalize_group_C
 *
 * \param A Matrix to scale
 * \param nrows Number of rows used
 * returns mineval Calculated min value
 * returns maxeval Calculated max value
 */
void *
bml_gershgorin_partial(
    const bml_matrix_t * A,
    const int nrows)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_gershgorin_partial_dense(A, nrows);
            break;
        case ellpack:
            return bml_gershgorin_partial_ellpack(A, nrows);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return NULL;
}
