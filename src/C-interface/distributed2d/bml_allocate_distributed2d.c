#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "../bml_logger.h"
#include "../bml_utilities.h"
#include "bml_allocate_distributed2d.h"
#include "bml_types_distributed2d.h"

#include <assert.h>
#include <math.h>

/* MPI communicator for all the distributed2d matrices */
static MPI_Comm s_comm = MPI_COMM_NULL;
static MPI_Comm r_comm = MPI_COMM_NULL;
static MPI_Comm c_comm = MPI_COMM_NULL;

void
bml_setcomm_distributed2d(
    MPI_Comm comm)
{
    // create new communicator
    int ntasks;
    MPI_Comm_size(comm, &ntasks);
    int mytask;
    MPI_Comm_rank(comm, &mytask);
    int p = bml_sqrtint(ntasks);
    if (p * p != ntasks)
    {
        LOG_ERROR("Invalid number of tasks. Must be an integer square.\n");
    }
    int dims[2] = { p, p };
    int periods[2] = { 1, 1 };
    int reorder = 0;
    MPI_Cart_create(comm, 2, dims, periods, reorder, &s_comm);

    // use seed based on task ID
    srand(13 * mytask + 17);
}

// set various fields in matrix struct
void
bml_setup_distributed2d(
    const int N,
    bml_matrix_distributed2d_t * A)
{
    assert(s_comm != MPI_COMM_NULL);

    A->comm = s_comm;

    int ntasks;
    MPI_Comm_size(s_comm, &ntasks);
    A->ntasks = ntasks;

    int mytask;
    MPI_Comm_rank(A->comm, &mytask);
    A->mpitask = mytask;

    int coords[2];
    MPI_Cart_coords(A->comm, A->mpitask, 2, coords);

    int p = bml_sqrtint(ntasks);
    A->nprows = p;
    A->npcols = p;
    A->myprow = coords[0];
    A->mypcol = coords[1];
    A->N = N;
    A->n = N / p;
    A->matrix_type = distributed2d;

    MPI_Comm_split(A->comm, A->myprow, A->mypcol, &A->row_comm);
    MPI_Comm_split(A->comm, A->mypcol, A->myprow, &A->col_comm);

    assert(A->n * p == N);
}

/** Deallocate a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void
bml_deallocate_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    assert(A != NULL);
    assert(A->matrix != NULL);
    bml_deallocate(&(A->matrix));
    bml_free_memory(A);
}

/** Clear a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void
bml_clear_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    bml_clear(A->matrix);
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
 *  \return The matrix.
 */
bml_matrix_distributed2d_t *
bml_zero_matrix_distributed2d(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M)
{
    assert(N > 0);
    assert(M > 0);

    bml_matrix_distributed2d_t *A =
        bml_allocate_memory(sizeof(bml_matrix_distributed2d_t));
    bml_setup_distributed2d(N, A);
    A->M = M;
    A->matrix_precision = matrix_precision;
    int m = M / bml_sqrtint(A->ntasks);
    A->matrix =
        bml_zero_matrix(matrix_type, matrix_precision, A->n, m, sequential);
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
 *  \return The matrix.
 */
bml_matrix_distributed2d_t *
bml_random_matrix_distributed2d(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M)
{
    assert(M > 0);

    bml_matrix_distributed2d_t *A =
        bml_allocate_memory(sizeof(bml_matrix_distributed2d_t));
    bml_setup_distributed2d(N, A);
    A->M = M;
    A->matrix_precision = matrix_precision;
    int m = M / bml_sqrtint(A->ntasks);
    A->matrix =
        bml_random_matrix(matrix_type, matrix_precision, A->n, m, sequential);
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
 *  \return The matrix.
 */
bml_matrix_distributed2d_t *
bml_identity_matrix_distributed2d(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M)
{
    assert(M > 0);

    bml_matrix_distributed2d_t *A =
        bml_allocate_memory(sizeof(bml_matrix_distributed2d_t));
    bml_setup_distributed2d(N, A);
    A->M = M;
    A->matrix_precision = matrix_precision;
    int m = M / bml_sqrtint(A->ntasks);
    A->matrix = (A->myprow == A->mypcol) ?
        bml_identity_matrix(matrix_type, matrix_precision, A->n, m,
                            sequential) : bml_zero_matrix(matrix_type,
                                                          matrix_precision,
                                                          A->n, m,
                                                          sequential);

    return A;
}
