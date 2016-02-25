#include "bml_logger.h"
#include "bml_trace.h"
#include "bml_norm_ellpack.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>

/** Calculate the sum of squares of the elements of a matrix.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix
 *  \return The sum of squares of A
 */
double
bml_sum_squares_ellpack(
    const bml_matrix_ellpack_t * A)
{

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_sum_squares_ellpack_single_real(A);
            break;
        case double_real:
            return bml_sum_squares_ellpack_double_real(A);
            break;
        case single_complex:
            return bml_sum_squares_ellpack_single_complex(A);
            break;
        case double_complex:
            return bml_sum_squares_ellpack_double_complex(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return 0;
}

/** Calculate the sum of squares of the elements of \alpha A + \beta B.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \param alpha Multiplier for A
 *  \param beta Multiplier for B
 *  \return The sum of squares of \alpha A + \beta B
 */
double
bml_sum_squares2_ellpack(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    const double alpha,
    const double beta)
{

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_sum_squares2_ellpack_single_real(A, B, alpha, beta);
            break;
        case double_real:
            return bml_sum_squares2_ellpack_double_real(A, B, alpha, beta);
            break;
        case single_complex:
            return bml_sum_squares2_ellpack_single_complex(A, B, alpha, beta);
            break;
        case double_complex:
            return bml_sum_squares2_ellpack_double_complex(A, B, alpha, beta);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return 0;
}

/* Calculate the Frobenius norm of a matrix.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \return The Frobenius norm of A
 */
double
bml_fnorm_ellpack(
    const bml_matrix_ellpack_t * A)
{

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_fnorm_ellpack_single_real(A);
            break;
        case double_real:
            return bml_fnorm_ellpack_double_real(A);
            break;
        case single_complex:
            return bml_fnorm_ellpack_single_complex(A);
            break;
        case double_complex:
            return bml_fnorm_ellpack_double_complex(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return 0;
}
