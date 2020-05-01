#include "../bml_logger.h"
#include "../bml_transpose.h"
#include "../bml_types.h"
#include "bml_transpose_csr.h"
#include "bml_types_csr.h"

#include <stdlib.h>
#include <string.h>

/** Transpose a matrix.
 *
 *  \ingroup transpose_group
 *
 *  \param A The matrix to be transposeed
 *  \return the transposeed A
 */
bml_matrix_csr_t *
bml_transpose_new_csr(
    bml_matrix_csr_t * A)
{
    bml_matrix_csr_t *B = NULL;

    switch (A->matrix_precision)
    {
        case single_real:
            B = bml_transpose_new_csr_single_real(A);
            break;
        case double_real:
            B = bml_transpose_new_csr_double_real(A);
            break;
        case single_complex:
            B = bml_transpose_new_csr_single_complex(A);
            break;
        case double_complex:
            B = bml_transpose_new_csr_double_complex(A);
            break;
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
bml_transpose_csr(
    bml_matrix_csr_t * A)
{

    switch (A->matrix_precision)
    {
        case single_real:
            bml_transpose_csr_single_real(A);
            break;
        case double_real:
            bml_transpose_csr_double_real(A);
            break;
        case single_complex:
            bml_transpose_csr_single_complex(A);
            break;
        case double_complex:
            bml_transpose_csr_double_complex(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
