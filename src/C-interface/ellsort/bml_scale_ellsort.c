#include "../bml_logger.h"
#include "../bml_scale.h"
#include "../bml_types.h"
#include "bml_scale_ellsort.h"
#include "bml_types_ellsort.h"

#include <stdlib.h>
#include <string.h>

/** Scale an ellsort matrix - result is a new matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \return A scale version of matrix A.
 */
bml_matrix_ellsort_t *
bml_scale_ellsort_new(
    void *scale_factor,
    bml_matrix_ellsort_t * A)
{
    bml_matrix_ellsort_t *B = NULL;

    switch (A->matrix_precision)
    {
        case single_real:
            B = bml_scale_ellsort_new_single_real(scale_factor, A);
            break;
        case double_real:
            B = bml_scale_ellsort_new_double_real(scale_factor, A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            B = bml_scale_ellsort_new_single_complex(scale_factor, A);
            break;
        case double_complex:
            B = bml_scale_ellsort_new_double_complex(scale_factor, A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return B;
}

/** Scale an ellsort matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \param B Scaled version of matrix A
 */
void
bml_scale_ellsort(
    void *scale_factor,
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_scale_ellsort_single_real(scale_factor, A, B);
            break;
        case double_real:
            bml_scale_ellsort_double_real(scale_factor, A, B);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_scale_ellsort_single_complex(scale_factor, A, B);
            break;
        case double_complex:
            bml_scale_ellsort_double_complex(scale_factor, A, B);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_scale_inplace_ellsort(
    void *scale_factor,
    bml_matrix_ellsort_t * A)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_scale_inplace_ellsort_single_real(scale_factor, A);
            break;
        case double_real:
            bml_scale_inplace_ellsort_double_real(scale_factor, A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_scale_inplace_ellsort_single_complex(scale_factor, A);
            break;
        case double_complex:
            bml_scale_inplace_ellsort_double_complex(scale_factor, A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
