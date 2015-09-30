#include "bml_logger.h"
#include "bml_multiply.h"
#include "bml_multiply_dense.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

/** Matrix multiply.
 *
 * C = alpha * A * B + beta * C
 *
 *  \ingroup multiply_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param C Matrix C
 *  \param alpha Scalar factor multiplied by A * B
 *  \param beta Scalar factor multiplied by C
 */
void
bml_multiply_dense(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const bml_matrix_dense_t * C,
    const double alpha,
    const double beta)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_multiply_dense_single_real(A, B, C, alpha, beta);
            break;
        case double_real:
            bml_multiply_dense_double_real(A, B, C, alpha, beta);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

/** Matrix multiply.
 *
 * X2 = X * X
 *
 *  \ingroup multiply_group
 *
 *  \param X Matrix X
 *  \param X2 Matrix X2
 */
void
bml_multiply_x2_dense(
    const bml_matrix_dense_t * X,
    const bml_matrix_dense_t * X2)
{
    switch (X->matrix_precision)
    {
        case single_real:
            bml_multiply_x2_dense_single_real(X, X2);
            break;
        case double_real:
            bml_multiply_x2_dense_double_real(X, X2);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
