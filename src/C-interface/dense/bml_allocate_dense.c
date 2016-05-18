#include "bml_allocate.h"
#include "bml_allocate_dense.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_dense.h"

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
    bml_deallocate_domain(A->domain);

    bml_free_memory(A->domain);
    bml_free_memory(A->matrix);
    bml_free_memory(A);
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
 *  \return The matrix.
 */
bml_matrix_dense_t *
bml_zero_matrix_dense(
    const bml_matrix_precision_t matrix_precision,
    const int N)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_zero_matrix_dense_single_real(N);
            break;
        case double_real:
            return bml_zero_matrix_dense_double_real(N);
            break;
        case single_complex:
            return bml_zero_matrix_dense_single_complex(N);
            break;
        case double_complex:
            return bml_zero_matrix_dense_double_complex(N);
            break;
        default:
            LOG_ERROR("unknown precision\n");
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
 * \return The matrix.
 */
bml_matrix_dense_t *
bml_banded_matrix_dense(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_banded_matrix_dense_single_real(N, M);
            break;
        case double_real:
            return bml_banded_matrix_dense_double_real(N, M);
            break;
        case single_complex:
            return bml_banded_matrix_dense_single_complex(N, M);
            break;
        case double_complex:
            return bml_banded_matrix_dense_double_complex(N, M);
            break;
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
 *  \return The matrix.
 */
bml_matrix_dense_t *
bml_random_matrix_dense(
    const bml_matrix_precision_t matrix_precision,
    const int N)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_random_matrix_dense_single_real(N);
            break;
        case double_real:
            return bml_random_matrix_dense_double_real(N);
            break;
        case single_complex:
            return bml_random_matrix_dense_single_complex(N);
            break;
        case double_complex:
            return bml_random_matrix_dense_double_complex(N);
            break;
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
 *  \return The matrix.
 */
bml_matrix_dense_t *
bml_identity_matrix_dense(
    const bml_matrix_precision_t matrix_precision,
    const int N)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_identity_matrix_dense_single_real(N);
            break;
        case double_real:
            return bml_identity_matrix_dense_double_real(N);
            break;
        case single_complex:
            return bml_identity_matrix_dense_single_complex(N);
            break;
        case double_complex:
            return bml_identity_matrix_dense_double_complex(N);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
