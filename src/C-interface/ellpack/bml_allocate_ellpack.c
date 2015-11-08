#include "bml_allocate.h"
#include "bml_allocate_ellpack.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

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
    bml_free_memory(A->value);
    bml_free_memory(A->index);
    bml_free_memory(A->nnz);
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
 *  \param M The number of non-zeroes per row.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *
bml_zero_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M)
{
    bml_matrix_ellpack_t *A = NULL;

    switch (matrix_precision)
    {
        case single_real:
            A = bml_zero_matrix_ellpack_single_real(N, M);
            break;
        case double_real:
            A = bml_zero_matrix_ellpack_double_real(N, M);
            break;
        case single_complex:
            A = bml_zero_matrix_ellpack_single_complex(N, M);
            break;
        case double_complex:
            A = bml_zero_matrix_ellpack_double_complex(N, M);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return A;
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
 *  \return The matrix.
 */
bml_matrix_ellpack_t *
bml_random_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M)
{
    bml_matrix_ellpack_t *A = NULL;

    switch (matrix_precision)
    {
        case single_real:
            A = bml_random_matrix_ellpack_single_real(N, M);
            break;
        case double_real:
            A = bml_random_matrix_ellpack_double_real(N, M);
            break;
        case single_complex:
            A = bml_random_matrix_ellpack_single_complex(N, M);
            break;
        case double_complex:
            A = bml_random_matrix_ellpack_double_complex(N, M);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return A;
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
 *  \return The matrix.
 */
bml_matrix_ellpack_t *
bml_identity_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M)
{
    bml_matrix_ellpack_t *A = NULL;

    switch (matrix_precision)
    {
        case single_real:
            A = bml_identity_matrix_ellpack_single_real(N, M);
            break;
        case double_real:
            A = bml_identity_matrix_ellpack_double_real(N, M);
            break;
        case single_complex:
            A = bml_identity_matrix_ellpack_single_complex(N, M);
            break;
        case double_complex:
            A = bml_identity_matrix_ellpack_double_complex(N, M);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return A;
}
