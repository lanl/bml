#include "../bml_logger.h"
#include "../bml_transpose.h"
#include "../bml_types.h"
#include "bml_transpose_ellsort.h"
#include "bml_types_ellsort.h"

#include <stdlib.h>
#include <string.h>

/** Transpose a matrix.
 *
 *  \ingroup transpose_group
 *
 *  \param A The matrix to be transposeed
 *  \return the transposeed A
 */
bml_matrix_ellsort_t *
bml_transpose_new_ellsort(
    bml_matrix_ellsort_t * A)
{
    bml_matrix_ellsort_t *B = NULL;

    switch (A->matrix_precision)
    {
        case single_real:
            B = bml_transpose_new_ellsort_single_real(A);
            break;
        case double_real:
            B = bml_transpose_new_ellsort_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            B = bml_transpose_new_ellsort_single_complex(A);
            break;
        case double_complex:
            B = bml_transpose_new_ellsort_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return B;
}

/** Transpose a matrix in place.
 *
 *  \ingroup transpose_group
 *
 *  \param A The matrix to be transposeed
 *  \return the transposeed A
 */
void
bml_transpose_ellsort(
    bml_matrix_ellsort_t * A)
{

    switch (A->matrix_precision)
    {
        case single_real:
            bml_transpose_ellsort_single_real(A);
            break;
        case double_real:
            bml_transpose_ellsort_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_transpose_ellsort_single_complex(A);
            break;
        case double_complex:
            bml_transpose_ellsort_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
