#include "bml_norm.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_norm_dense.h"
#include "ellpack/bml_norm_ellpack.h"

#include <stdlib.h>

/** Calculate the sum of squares of all the elements of a matrix.
 *
 * \ingroup norm_group_C
 *
 * \param A Matrix A
 * \return sum of squares of all elements in A
 */
double
bml_sum_squares(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_sum_squares_dense(A);
            break;
        case ellpack:
            return bml_sum_squares_ellpack(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return 0;
}

/** Calculate sum of squares of all the elements of 
 * \alpha A + \beta B
 * \ingroup norm_group_C
 *
 * \param A Matrix
 * \param B Matrix
 * \param alpha Multiplier for matrix A
 * \param beta Multiplier for matrix B
 * \param threshold Threshold
 * \return sum of squares of alpha * A + beta * B 
 */
double
bml_sum_squares2(
    const bml_matrix_t * A,
    const bml_matrix_t * B,
    const double alpha,
    const double beta,
    const double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_sum_squares2_dense(A, B, alpha, beta, threshold);
            break;
        case ellpack:
            return bml_sum_squares2_ellpack(A, B, alpha, beta, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return 0;
}

/** Calculate the Frobenius norm of a matrix.
 *
 * \ingroup norm_group_C
 *
 * \param A Matrix A
 * \return Frobenius norm of Matrix A
 */
double
bml_fnorm(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_fnorm_dense(A);
            break;
        case ellpack:
            return bml_fnorm_ellpack(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return 0;
}

