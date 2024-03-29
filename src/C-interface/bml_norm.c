#include "bml_norm.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_norm_dense.h"
#include "ellpack/bml_norm_ellpack.h"
#include "ellblock/bml_norm_ellblock.h"
#include "csr/bml_norm_csr.h"
#ifdef BML_USE_MPI
#include "distributed2d/bml_norm_distributed2d.h"
#endif

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
    bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_sum_squares_dense(A);
            break;
        case ellpack:
            return bml_sum_squares_ellpack(A);
            break;
        case ellblock:
            return bml_sum_squares_ellblock(A);
            break;
        case csr:
            return bml_sum_squares_csr(A);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            return bml_sum_squares_distributed2d(A);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return 0;
}

/** Calculate the sum of squares of all the elements of a matrix.
 *
 * \ingroup norm_group_C
 *
 * \param A Matrix A
 * \param core_pos Core rows in A
 * \param core_size Number of core rows
 * \return sum of squares of all elements in A
 */
double
bml_sum_squares_submatrix(
    bml_matrix_t * A,
    int core_size)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_sum_squares_submatrix_dense(A, core_size);
            break;
        case ellpack:
            return bml_sum_squares_submatrix_ellpack(A, core_size);
            break;
        case csr:
            return bml_sum_squares_submatrix_csr(A, core_size);
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
    bml_matrix_t * A,
    bml_matrix_t * B,
    double alpha,
    double beta,
    double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_sum_squares2_dense(A, B, alpha, beta, threshold);
            break;
        case ellpack:
            return bml_sum_squares2_ellpack(A, B, alpha, beta, threshold);
            break;
        case ellblock:
            return bml_sum_squares2_ellblock(A, B, alpha, beta, threshold);
            break;
        case csr:
            return bml_sum_squares2_csr(A, B, alpha, beta, threshold);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            return bml_sum_squares2_distributed2d(A, B, alpha, beta,
                                                  threshold);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return 0;
}

/** Calculate sum of all the elements of
 * \alpha A(i,j) * B(i,j)
 * \ingroup norm_group_C
 *
 * \param A Matrix
 * \param B Matrix
 * \param alpha Multiplier for matrix A
 * \param threshold Threshold
 * \return sum of squares of alpha * A + beta * B
 */
double
bml_sum_AB(
    bml_matrix_t * A,
    bml_matrix_t * B,
    double alpha,
    double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_sum_AB_dense(A, B, alpha, threshold);
            break;
        case ellpack:
            return bml_sum_AB_ellpack(A, B, alpha, threshold);
            break;
        case ellblock:
            return bml_sum_AB_ellblock(A, B, alpha, threshold);
            break;
        case csr:
            return bml_sum_AB_csr(A, B, alpha, threshold);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            return bml_sum_AB_distributed2d(A, B, alpha, threshold);
            break;
#endif
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
    bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_fnorm_dense(A);
            break;
        case ellpack:
            return bml_fnorm_ellpack(A);
            break;
        case ellblock:
            return bml_fnorm_ellblock(A);
            break;
        case csr:
            return bml_fnorm_csr(A);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            return bml_fnorm_distributed2d(A);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return 0;
}

/** Calculate the Frobenius norm of 2 matrices.
 *
 * \ingroup norm_group_C
 *
 * \param A Matrix A
 * \param B Matrix B
 * \return Frobenius norm of Matrix A
 */
double
bml_fnorm2(
    bml_matrix_t * A,
    bml_matrix_t * B)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_fnorm2_dense(A, B);
            break;
        case ellpack:
            return bml_fnorm2_ellpack(A, B);
            break;
        case csr:
            return bml_fnorm2_csr(A, B);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return 0;
}
