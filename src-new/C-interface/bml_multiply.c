#include "bml_multiply.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_multiply_dense.h"
#include "ellpack/bml_multiply_ellpack.h"

#include <stdlib.h>

/** Matrix multiply.
 *
 * C = alpha * A * B + beat * C
 *
 * \ingroup multiply_group_C
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param alpha Scalar factor that multiplies A * B
 * \param beta Scalar factor that multiplies C
 * \param threshold Threshold for multiplication
 */
void
bml_multiply(
    const bml_matrix_t * A,
    const bml_matrix_t * B,
    const bml_matrix_t * C,
    const double alpha,
    const double beta,
    const double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_multiply_dense(A, B, C, alpha, beta);
            break;
        case ellpack:
            bml_multiply_ellpack(A, B, C, alpha, beta, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Matrix multiply.
 *
 * X2 = X * X
 *
 * \ingroup multiply_group_C
 *
 * \param X Matrix X
 * \param X2 MatrixX2 
 * \param threshold Threshold for multiplication
 */
void
bml_multiply_x2(
    const bml_matrix_t * X,
    const bml_matrix_t * X2,
    const double threshold)
{
    switch (bml_get_type(X))
    {
        case dense:
            bml_multiply_x2_dense(X, X2);
            break;
        case ellpack:
            bml_multiply_x2_ellpack(X, X2, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Matrix multiply.
 *
 * C = A * B
 *
 * \ingroup multiply_group_C
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param threshold Threshold for multiplication
 */
void
bml_multiply_AB(
    const bml_matrix_t * A,
    const bml_matrix_t * B,
    const bml_matrix_t * C,
    const double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_multiply_AB_dense(A, B, C);
            break;
        case ellpack:
            bml_multiply_AB_ellpack(A, B, C, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

