#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_normalize.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_normalize_dense.h"
#include "bml_types_dense.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

void *
bml_accumulate_offdiag_dense(
    bml_matrix_dense_t * A,
    int flag)
{
    switch (A->matrix_precision)
    {
        case single_real:
            return bml_accumulate_offdiag_dense_single_real(A, flag);
            break;
        case double_real:
            return bml_accumulate_offdiag_dense_double_real(A, flag);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_accumulate_offdiag_dense_single_complex(A, flag);
            break;
        case double_complex:
            return bml_accumulate_offdiag_dense_double_complex(A, flag);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            return NULL;
            break;
    }
}

/** Normalize dense matrix given Gershgorin bounds.
 *
 *  \ingroup normalize_group
 *
 *  \param A The matrix
 *  \param mineval Calculated min value
 *  \param maxeval Calculated max value
 */
void
bml_normalize_dense(
    bml_matrix_dense_t * A,
    double mineval,
    double maxeval)
{
    assert(A != NULL);

    switch (A->matrix_precision)
    {
        case single_real:
            bml_normalize_dense_single_real(A, mineval, maxeval);
            break;
        case double_real:
            bml_normalize_dense_double_real(A, mineval, maxeval);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_normalize_dense_single_complex(A, mineval, maxeval);
            break;
        case double_complex:
            bml_normalize_dense_double_complex(A, mineval, maxeval);
            break;
#endif
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
 *  returns mineval Calculated min value
 *  returns maxeval Calculated max value
 */
void *
bml_gershgorin_dense(
    bml_matrix_dense_t * A)
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
#ifdef BML_COMPLEX
        case single_complex:
            return bml_gershgorin_dense_single_complex(A);
            break;
        case double_complex:
            return bml_gershgorin_dense_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}

/** Calculate Gershgorin bounds for a partial dense matrix.
 *
 *  \ingroup normalize_group
 *
 *  \param A The matrix
 *  \param nrows Number of rows used
 *  returns mineval Calculated min value
 *  returns maxeval Calculated max value
 */
void *
bml_gershgorin_partial_dense(
    bml_matrix_dense_t * A,
    int nrows)
{
    assert(A != NULL);

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_gershgorin_partial_dense_single_real(A, nrows);
            break;
        case double_real:
            return bml_gershgorin_partial_dense_double_real(A, nrows);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_gershgorin_partial_dense_single_complex(A, nrows);
            break;
        case double_complex:
            return bml_gershgorin_partial_dense_double_complex(A, nrows);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
