#include "../bml_add.h"
#include "../bml_logger.h"
#include "../bml_types.h"
#include "bml_add_ellblock.h"
#include "bml_types_ellblock.h"

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
bml_add_ellblock(
    bml_matrix_ellblock_t * A,
    bml_matrix_ellblock_t * B,
    double alpha,
    double beta,
    double threshold)
{
    switch (B->matrix_precision)
    {
        case single_real:
            bml_add_ellblock_single_real(A, B, alpha, beta, threshold);
            break;
        case double_real:
            bml_add_ellblock_double_real(A, B, alpha, beta, threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_add_ellblock_single_complex(A, B, alpha, beta, threshold);
            break;
        case double_complex:
            bml_add_ellblock_double_complex(A, B, alpha, beta, threshold);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
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
bml_add_norm_ellblock(
    bml_matrix_ellblock_t * A,
    bml_matrix_ellblock_t * B,
    double alpha,
    double beta,
    double threshold)
{
    double trnorm = 0.0;

    switch (B->matrix_precision)
    {
        case single_real:
            trnorm =
                bml_add_norm_ellblock_single_real(A, B, alpha, beta,
                                                  threshold);
            break;
        case double_real:
            trnorm =
                bml_add_norm_ellblock_double_real(A, B, alpha, beta,
                                                  threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            trnorm =
                bml_add_norm_ellblock_single_complex(A, B, alpha, beta,
                                                     threshold);
            break;
        case double_complex:
            trnorm =
                bml_add_norm_ellblock_double_complex(A, B, alpha, beta,
                                                     threshold);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
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
bml_add_identity_ellblock(
    bml_matrix_ellblock_t * A,
    double beta,
    double threshold)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_add_identity_ellblock_single_real(A, beta, threshold);
            break;
        case double_real:
            bml_add_identity_ellblock_double_real(A, beta, threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_add_identity_ellblock_single_complex(A, beta, threshold);
            break;
        case double_complex:
            bml_add_identity_ellblock_double_complex(A, beta, threshold);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
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
bml_scale_add_identity_ellblock(
    bml_matrix_ellblock_t * A,
    double alpha,
    double beta,
    double threshold)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_scale_add_identity_ellblock_single_real(A, alpha, beta,
                                                        threshold);
            break;
        case double_real:
            bml_scale_add_identity_ellblock_double_real(A, alpha, beta,
                                                        threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_scale_add_identity_ellblock_single_complex(A, alpha, beta,
                                                           threshold);
            break;
        case double_complex:
            bml_scale_add_identity_ellblock_double_complex(A, alpha, beta,
                                                           threshold);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
