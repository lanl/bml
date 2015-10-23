#include "bml_allocate.h"
#include "bml_allocate_ellpack.h"
#include "bml_convert.h"
#include "bml_convert_ellpack.h"
#include "bml_logger.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group
 *
 * \param N The number of rows/columns
 * \param matrix_precision The real precision
 * \param A The dense matrix
 * \param threshold The matrix element magnited threshold
 * \return The bml matrix
 */
bml_matrix_ellpack_t *
bml_convert_from_dense_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const void *A,
    const double threshold,
    const int M)
{
    bml_matrix_ellpack_t *A_bml = NULL;

    switch (matrix_precision)
    {
        case single_real:
            A_bml = bml_convert_from_dense_ellpack_single_real(N, A, threshold, M);
            break;
        case double_real:
            A_bml = bml_convert_from_dense_ellpack_double_real(N, A, threshold, M);
            break;
        case single_complex:
            A_bml = bml_convert_from_dense_ellpack_single_complex(N, A, threshold, M);
            break;
        case double_complex:
            A_bml = bml_convert_from_dense_ellpack_double_complex(N, A, threshold, M);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return A_bml;
}

/** Convert a bml matrix into a dense matrix.
 *
 * \ingroup convert_group
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
void *
bml_convert_to_dense_ellpack(
    const bml_matrix_ellpack_t * A)
{
    float *A_float = NULL;
    double *A_double = NULL;

    switch (A->matrix_precision)
    {
        case single_real:
            A_float = bml_convert_to_dense_ellpack_single_real(A);
            return A_float;
            break;
        case double_real:
            A_double = bml_convert_to_dense_ellpack_double_real(A);
            return A_double;
            break;
        case single_complex:
            A_float = bml_convert_to_dense_ellpack_single_complex(A);
            return A_float;
            break;
        case double_complex:
            A_double = bml_convert_to_dense_ellpack_double_complex(A);
            return A_double;
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
