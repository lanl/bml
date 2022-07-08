#include "../bml_logger.h"
#include "../bml_trace.h"
#include "../bml_types.h"
#include "bml_norm_ellsort.h"
#include "bml_types_ellsort.h"

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
bml_sum_squares_ellsort(
    bml_matrix_ellsort_t * A)
{

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_sum_squares_ellsort_single_real(A);
            break;
        case double_real:
            return bml_sum_squares_ellsort_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_sum_squares_ellsort_single_complex(A);
            break;
        case double_complex:
            return bml_sum_squares_ellsort_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return 0;
}

/** Calculate the sum of squares of the core elements in submatrix A.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix
 *  \param core_pos Core rows in submatrix
 *  \param core_size Number of core rows
 *  \return Sum of squares of all elements in A
 */
double
bml_sum_squares_submatrix_ellsort(
    bml_matrix_ellsort_t * A,
    int core_size)
{
    switch (A->matrix_precision)
    {
        case single_real:
            return bml_sum_squares_submatrix_ellsort_single_real(A,
                                                                 core_size);
            break;
        case double_real:
            return bml_sum_squares_submatrix_ellsort_double_real(A,
                                                                 core_size);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_sum_squares_submatrix_ellsort_single_complex(A,
                                                                    core_size);
            break;
        case double_complex:
            return bml_sum_squares_submatrix_ellsort_double_complex(A,
                                                                    core_size);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return 0;
}

/** Calculate the sum of the elements of \alpha A(i,j) * B(i,j).
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \param alpha Multiplier for A
 *  \param threshold Threshold
 *  \return The sum of squares of \alpha A(i,j) * B(i,j)
 */
double
bml_sum_AB_ellsort(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B,
    double alpha,
    double threshold)
{

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_sum_AB_ellsort_single_real(A, B, alpha, threshold);
            break;
        case double_real:
            return bml_sum_AB_ellsort_double_real(A, B, alpha, threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_sum_AB_ellsort_single_complex(A, B, alpha, threshold);
            break;
        case double_complex:
            return bml_sum_AB_ellsort_double_complex(A, B, alpha, threshold);
            break;
#endif
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
 *  \param threshold Threshold
 *  \return The sum of squares of \alpha A + \beta B
 */
double
bml_sum_squares2_ellsort(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B,
    double alpha,
    double beta,
    double threshold)
{

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_sum_squares2_ellsort_single_real(A, B, alpha, beta,
                                                        threshold);
            break;
        case double_real:
            return bml_sum_squares2_ellsort_double_real(A, B, alpha, beta,
                                                        threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_sum_squares2_ellsort_single_complex(A, B, alpha, beta,
                                                           threshold);
            break;
        case double_complex:
            return bml_sum_squares2_ellsort_double_complex(A, B, alpha, beta,
                                                           threshold);
            break;
#endif
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
bml_fnorm_ellsort(
    bml_matrix_ellsort_t * A)
{

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_fnorm_ellsort_single_real(A);
            break;
        case double_real:
            return bml_fnorm_ellsort_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_fnorm_ellsort_single_complex(A);
            break;
        case double_complex:
            return bml_fnorm_ellsort_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return 0;
}

/* Calculate the Frobenius norm of 2 matrices.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \return The Frobenius norm of A-B
 */
double
bml_fnorm2_ellsort(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B)
{

    switch (A->matrix_precision)
    {
        case single_real:
            return bml_fnorm2_ellsort_single_real(A, B);
            break;
        case double_real:
            return bml_fnorm2_ellsort_double_real(A, B);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_fnorm2_ellsort_single_complex(A, B);
            break;
        case double_complex:
            return bml_fnorm2_ellsort_double_complex(A, B);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return 0;
}
