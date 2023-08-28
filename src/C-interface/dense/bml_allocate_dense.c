#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "../bml_domain.h"
#include "bml_allocate_dense.h"
#include "bml_types_dense.h"


#ifdef BML_USE_MAGMA
static magma_queue_t queue;
magma_queue_t
bml_queue(
    )
{
    return queue;
}

void
bml_queue_create(
    int device)
{
    static int queueset = 0;
    if (queueset == 0)
    {
        LOG_INFO("magma_queue_create\n");
        magma_queue_create(device, &queue);
        queueset = 1;
    }
}
#endif

#ifdef MKL_GPU
#include "stdio.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#endif

/** Deallocate a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void
bml_deallocate_dense(
    bml_matrix_dense_t * A)
{
    switch (A->matrix_precision)
    {
        case single_real:
            return bml_deallocate_dense_single_real(A);
            break;
        case double_real:
            return bml_deallocate_dense_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_deallocate_dense_single_complex(A);
            break;
        case double_complex:
            return bml_deallocate_dense_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision (%d)\n", A->matrix_precision);
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
bml_clear_dense(
    bml_matrix_dense_t * A)
{
    switch (A->matrix_precision)
    {
        case single_real:
            return bml_clear_dense_single_real(A);
            break;
        case double_real:
            return bml_clear_dense_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_clear_dense_single_complex(A);
            break;
        case double_complex:
            return bml_clear_dense_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision (%d)\n", A->matrix_precision);
            break;
    }
}

/** Allocate an uninitialized matrix.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param matrix_dimension The matrix size.
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_dense_t *
bml_noinit_matrix_dense(
    bml_matrix_precision_t matrix_precision,
    bml_matrix_dimension_t matrix_dimension,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_noinit_matrix_dense_single_real(matrix_dimension,
                                                       distrib_mode);
            break;
        case double_real:
            return bml_noinit_matrix_dense_double_real(matrix_dimension,
                                                       distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_noinit_matrix_dense_single_complex(matrix_dimension,
                                                          distrib_mode);
            break;
        case double_complex:
            return bml_noinit_matrix_dense_double_complex(matrix_dimension,
                                                          distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision (%d)\n", matrix_precision);
            break;
    }
    return NULL;
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
bml_matrix_dense_t *
bml_zero_matrix_dense(
    bml_matrix_precision_t matrix_precision,
    bml_matrix_dimension_t matrix_dimension,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_zero_matrix_dense_single_real(matrix_dimension,
                                                     distrib_mode);
            break;
        case double_real:
            return bml_zero_matrix_dense_double_real(matrix_dimension,
                                                     distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_zero_matrix_dense_single_complex(matrix_dimension,
                                                        distrib_mode);
            break;
        case double_complex:
            return bml_zero_matrix_dense_double_complex(matrix_dimension,
                                                        distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision (%d)\n", matrix_precision);
            break;
    }
    return NULL;
}

/** Allocate a banded matrix.
 *
 * Note that the matrix \f$ a \f$ will be newly allocated. If it is
 * already allocated then the matrix will be deallocated in the
 * process.
 *
 * \ingroup allocate_group
 *
 * \param matrix_precision The precision of the matrix. The default
 * is double precision.
 * \param N The matrix size.
 * \param M The bandwidth.
 * \param distrib_mode The distribution mode.
 * \return The matrix.
 */
bml_matrix_dense_t *
bml_banded_matrix_dense(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_banded_matrix_dense_single_real(N, M, distrib_mode);
            break;
        case double_real:
            return bml_banded_matrix_dense_double_real(N, M, distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_banded_matrix_dense_single_complex(N, M, distrib_mode);
            break;
        case double_complex:
            return bml_banded_matrix_dense_double_complex(N, M, distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision (%d)\n", matrix_precision);
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
 *  \param distrib_mode The distribution mode.
 *  \return The matrix.
 */
bml_matrix_dense_t *
bml_random_matrix_dense(
    bml_matrix_precision_t matrix_precision,
    int N,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_random_matrix_dense_single_real(N, distrib_mode);
            break;
        case double_real:
            return bml_random_matrix_dense_double_real(N, distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_random_matrix_dense_single_complex(N, distrib_mode);
            break;
        case double_complex:
            return bml_random_matrix_dense_double_complex(N, distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision (%d)\n", matrix_precision);
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
 *  \param distrib_mode The distribution mode
 *  \return The matrix.
 */
bml_matrix_dense_t *
bml_identity_matrix_dense(
    bml_matrix_precision_t matrix_precision,
    int N,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_identity_matrix_dense_single_real(N, distrib_mode);
            break;
        case double_real:
            return bml_identity_matrix_dense_double_real(N, distrib_mode);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_identity_matrix_dense_single_complex(N, distrib_mode);
            break;
        case double_complex:
            return bml_identity_matrix_dense_double_complex(N, distrib_mode);
            break;
#endif
        default:
            LOG_ERROR("unknown precision (%d)\n", matrix_precision);
            break;
    }
    return NULL;
}

/** Update the dense matrix domain.
 *
 * \ingroup allocate_group
 *
 * \param A Matrix with domain
 * \param localPartMin first part on each rank
 * \param localPartMin last part on each rank
 * \param nnodesInPart number of nodes per part
 */
void
bml_update_domain_dense(
    bml_matrix_dense_t * A,
    int *localPartMin,
    int *localPartMax,
    int *nnodesInPart)
{
    bml_domain_t *A_domain = A->domain;

    bml_update_domain(A_domain, localPartMin, localPartMax, nnodesInPart);
}
