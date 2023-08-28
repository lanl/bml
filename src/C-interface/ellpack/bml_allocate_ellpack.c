#include "../bml_allocate.h"
#include "../bml_domain.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_allocate_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdio.h>

/** Deallocate a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void
bml_deallocate_ellpack(
    bml_matrix_ellpack_t * A)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_deallocate_ellpack_single_real(A);
            break;
        case double_real:
            bml_deallocate_ellpack_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_deallocate_ellpack_single_complex(A);
            break;
        case double_complex:
            bml_deallocate_ellpack_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

/** Clear a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void
bml_clear_ellpack(
    bml_matrix_ellpack_t * A)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_clear_ellpack_single_real(A);
            break;
        case double_real:
            bml_clear_ellpack_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_clear_ellpack_single_complex(A);
            break;
        case double_complex:
            bml_clear_ellpack_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

/** Allocate the zero matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param matrix_dimension The matrix size.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *
bml_noinit_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    bml_matrix_dimension_t matrix_dimension,
    bml_distribution_mode_t distrib_mode)
{
    bml_matrix_ellpack_t *A = NULL;

    switch (matrix_precision)
    {
        case single_real:
            A = bml_noinit_matrix_ellpack_single_real(matrix_dimension,
                                                      distrib_mode);
            break;
        case double_real:
            A = bml_noinit_matrix_ellpack_double_real(matrix_dimension,
                                                      distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            A = bml_noinit_matrix_ellpack_single_complex(matrix_dimension,
                                                         distrib_mode);
            break;
        case double_complex:
            A = bml_noinit_matrix_ellpack_double_complex(matrix_dimension,
                                                         distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return A;
}

/** Allocate the zero matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *
bml_zero_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    bml_matrix_ellpack_t *A = NULL;

    switch (matrix_precision)
    {
        case single_real:
            A = bml_zero_matrix_ellpack_single_real(N, M, distrib_mode);
            break;
        case double_real:
            A = bml_zero_matrix_ellpack_double_real(N, M, distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            A = bml_zero_matrix_ellpack_single_complex(N, M, distrib_mode);
            break;
        case double_complex:
            A = bml_zero_matrix_ellpack_double_complex(N, M, distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return A;
}

/** Allocate a banded random matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *
bml_banded_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_banded_matrix_ellpack_single_real(N, M, distrib_mode);
            break;
        case double_real:
            return bml_banded_matrix_ellpack_double_real(N, M, distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_banded_matrix_ellpack_single_complex(N, M,
                                                            distrib_mode);
            break;
        case double_complex:
            return bml_banded_matrix_ellpack_double_complex(N, M,
                                                            distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}

/** Allocate a random matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *
bml_random_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_random_matrix_ellpack_single_real(N, M, distrib_mode);
            break;
        case double_real:
            return bml_random_matrix_ellpack_double_real(N, M, distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_random_matrix_ellpack_single_complex(N, M,
                                                            distrib_mode);
            break;
        case double_complex:
            return bml_random_matrix_ellpack_double_complex(N, M,
                                                            distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}

/** Allocate the identity matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *
bml_identity_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_identity_matrix_ellpack_single_real(N, M,
                                                           distrib_mode);
            break;
        case double_real:
            return bml_identity_matrix_ellpack_double_real(N, M,
                                                           distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_identity_matrix_ellpack_single_complex(N, M,
                                                              distrib_mode);
            break;
        case double_complex:
            return bml_identity_matrix_ellpack_double_complex(N, M,
                                                              distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}

/** Update the ellpack matrix domain.
 *
 * \ingroup allocate_group
 *
 * \param A Matrix with domain
 * \param localPartMin first part on each rank
 * \param localPartMin last part on each rank
 * \param nnodesInPart number of nodes per part
 */
void
bml_update_domain_ellpack(
    bml_matrix_ellpack_t * A,
    int *localPartMin,
    int *localPartMax,
    int *nnodesInPart)
{
    bml_domain_t *A_domain = A->domain;

    bml_update_domain(A_domain, localPartMin, localPartMax, nnodesInPart);
}
