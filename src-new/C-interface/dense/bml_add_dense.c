#include "bml_add.h"
#include "bml_add_dense.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

/** Matrix addition.
 *
 * A = alpha * A + beta * B
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param alpha Scalar factor multiplied by A
 *  \param beta Scalar factor multiplied by B
 */
void
bml_add_dense(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta)
{
    switch (A->matrix_precision)
    {
    case single_real:
        bml_add_dense_single_real(A, B, alpha, beta);
        break;
    case double_real:
        bml_add_dense_double_real(A, B, alpha, beta);
        break;
    default:
        LOG_ERROR("unknown precision\n");
        break;
    }
}

/** Matrix addition.
 *
 * A = A + beta * I
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param beta Scalar factor multiplied by A
 */
void
bml_add_identity_dense(
    const bml_matrix_dense_t * A,
    const double beta)
{
    switch (A->matrix_precision)
    {
    case single_real:
        bml_add_identity_dense_single_real(A, beta);
        break;
    case double_real:
        bml_add_identity_dense_double_real(A, beta);
        break;
    default:
        LOG_ERROR("unknown precision\n");
        break;
    }
}
