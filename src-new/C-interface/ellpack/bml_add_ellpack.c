#include "bml_add.h"
#include "bml_add_ellpack.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

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
 *  \param threshold Threshold for matrix addition
 */
void
bml_add_ellpack(
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    const double alpha,
    const double beta,
    const double threshold)
{
    switch (B->matrix_precision)
    {
        case single_real:
            bml_add_ellpack_single_real(A, B, alpha, beta, threshold);
            break;
        case double_real:
            bml_add_ellpack_double_real(A, B, alpha, beta, threshold);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

/** Matrix addition.
 *
 *  A = A + beta * I
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param beta Scalar factor multiplied by A
 *  \param threshold Threshold for matrix addition
 */
void
bml_add_identity_ellpack(
    const bml_matrix_ellpack_t * A,
    const double beta,
    const double threshold)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_add_identity_ellpack_single_real(A, beta, threshold);
            break;
        case double_real:
            bml_add_identity_ellpack_double_real(A, beta, threshold);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
