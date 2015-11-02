#include "bml_add.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_add_dense.h"
#include "ellpack/bml_add_ellpack.h"

#include <stdlib.h>

/** Matrix addition.
 *
 * \f$ A \leftarrow \alpha A + \beta B \f$
 *
 * \ingroup add_group_C
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param alpha Scalar factor multiplied by A
 * \param beta Scalar factor multiplied by B
 * \param threshold Threshold for matrix addition
 */
void
bml_add(
    const bml_matrix_t * A,
    const bml_matrix_t * B,
    const double alpha,
    const double beta,
    const double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_add_dense(A, B, alpha, beta);
            break;
        case ellpack:
            bml_add_ellpack(A, B, alpha, beta, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Matrix addition.
 *
 * \f$ A \leftarrow A + \beta \mathrm{Id} \f$
 *
 * \ingroup add_group_C
 *
 * \param A Matrix A
 * \param beta Scalar factor multiplied by A
 * \param threshold Threshold for matrix addition
 */
void
bml_add_identity(
    const bml_matrix_t * A,
    const double beta,
    const double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_add_identity_dense(A, beta);
            break;
        case ellpack:
            bml_add_identity_ellpack(A, beta, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
