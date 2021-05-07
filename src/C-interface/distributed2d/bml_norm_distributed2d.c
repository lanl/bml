#include "../bml_logger.h"
#include "../bml_types.h"
#include "../bml_norm.h"
#include "../bml_parallel.h"
#include "bml_norm_distributed2d.h"
#include "bml_types_distributed2d.h"

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Calculate the sum of squares of the elements in matrix A.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix
 *  \return Sum of squares of all elements in A
 */
double
bml_sum_squares_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    double norm = bml_sum_squares(A->matrix);

    bml_sumRealReduce(&norm);

    return norm;
}

/** Calculate the sum of elements in
 * \alpha A(i,j) * B(i,j).
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \param alpha Multiplier for matrix A
 *  \param threshold Threshold
 *  \return The sum of squares of all elements of \alpha A(i,j) * B(i,j)
 */
double
bml_sum_AB_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B,
    double alpha,
    double threshold)
{
    double norm = bml_sum_AB(A->matrix, B->matrix, alpha, threshold);

    bml_sumRealReduce(&norm);

    return norm;
}

/** Calculate the sum of squares of elements in
 * \alpha A + \beta B.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \param alpha Multiplier for matrix A
 *  \param beta Multiplier for matrix B
 *  \param threshold Threshold
 *  \return The sum of squares of all elements of \alpha A + \beta B
 */
double
bml_sum_squares2_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B,
    double alpha,
    double beta,
    double threshold)
{
    double norm =
        bml_sum_squares2(A->matrix, B->matrix, alpha, beta, threshold);
    norm = norm * norm;

    bml_sumRealReduce(&norm);

    return sqrt(norm);
}

/** Calculate the Fobenius norm of matrix A.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix
 *  \return Frobenius norm of A
 */
double
bml_fnorm_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    double norm = bml_fnorm(A->matrix);

    bml_sumRealReduce(&norm);

    return norm;
}

/** Calculate the Fobenius norm of the difference between 2 matrices.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \return Frobenius norm of A-B
 */
double
bml_fnorm2_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B)
{
    double norm = bml_fnorm2(A->matrix, B->matrix);
    norm = norm * norm;

    bml_sumRealReduce(&norm);

    return sqrt(norm);
}
