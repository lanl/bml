#include "../bml_add.h"
#include "../bml_logger.h"
#include "../bml_types.h"
#include "../bml_introspection.h"
#include "../bml_parallel.h"
#include "bml_add_distributed2d.h"
#include "bml_types_distributed2d.h"

#include <stdlib.h>
#include <string.h>

/** Matrix addition.
 *
 * \f$ A = \alpha A + \beta B \f$
 *
 * \ingroup add_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param alpha Scalar factor multiplied by A
 * \param beta Scalar factor multiplied by B
 * \param threshold Threshold for matrix addition
 */
void
bml_add_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B,
    double alpha,
    double beta,
    double threshold)
{
    bml_add(A->matrix, B->matrix, alpha, beta, threshold);
}

/** Matrix addition and TrNorm calculation.
 *
 * \f$ A = \alpha A + \beta B \f$
 *
 * \ingroup add_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param alpha Scalar factor multiplied by A
 * \param beta Scalar factor multiplied by B
 * \param threshold Threshold for matrix addition
 */
double
bml_add_norm_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B,
    double alpha,
    double beta,
    double threshold)
{
    double trnorm = bml_add_norm(A->matrix, B->matrix, alpha, beta,
                                 threshold);
    bml_sumRealReduce(&trnorm);

    return trnorm;
}

/** Matrix addition.
 *
 * \f$ A = A + \beta \mathrm{Id} \f$
 *
 * \ingroup add_group
 *
 * \param A Matrix A
 * \param beta Scalar factor multiplied by A
 * \param threshold Threshold for matrix addition
 */
void
bml_add_identity_distributed2d(
    bml_matrix_distributed2d_t * A,
    double beta,
    double threshold)
{
    bml_add_identity(A->matrix, beta, threshold);
}

/** Matrix addition.
 *
 * \f$ A = A + \beta \mathrm{Id} \f$
 *
 * \ingroup add_group
 *
 * \param A Matrix A
 * \param beta Scalar factor multiplied by A
 * \param threshold Threshold for matrix addition
 */
void
bml_scale_add_identity_distributed2d(
    bml_matrix_distributed2d_t * A,
    double alpha,
    double beta,
    double threshold)
{
    bml_scale_add_identity(A->matrix, alpha, beta, threshold);
}
