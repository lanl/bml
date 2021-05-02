#include "../bml_logger.h"
#include "../bml_transpose.h"
#include "../bml_types.h"
#include "../bml_parallel.h"
#include "bml_allocate_distributed2d.h"
#include "bml_transpose_distributed2d.h"
#include "bml_types_distributed2d.h"
#include "bml_copy_distributed2d.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

/** Transpose a matrix.
 *
 *  \ingroup transpose_group
 *
 *  \param A The matrix to be transposeed
 *  \return the transposeed A
 */
bml_matrix_distributed2d_t *
bml_transpose_new_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    assert(A->M > 0);

    bml_matrix_distributed2d_t *B = bml_copy_distributed2d_new(A);
    assert(B != NULL);

    if (A->myprow != A->mypcol)
    {
        assert(A->mpitask == A->myprow * A->npcols + A->mypcol);
        int remote_task = A->mypcol * A->npcols + A->myprow;
        bml_mpi_send(A->matrix, remote_task, A->comm);
        bml_mpi_recv(B->matrix, remote_task, A->comm);
    }

    bml_transpose(B->matrix);

    return B;
}

/** Transpose a matrix in place.
 *
 *  \ingroup transpose_group
 *
 *  \param A The matrix to be transposeed
 *  \return the transposeed A
 */
void
bml_transpose_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    assert(A->M > 0);

    bml_matrix_distributed2d_t *B = bml_copy_distributed2d_new(A);

    if (A->myprow != A->mypcol)
    {
        assert(A->mpitask == A->myprow * A->npcols + A->mypcol);
        int remote_task = A->mypcol * A->npcols + A->myprow;
        bml_mpi_send(B->matrix, remote_task, A->comm);
        bml_mpi_recv(A->matrix, remote_task, A->comm);
    }

    bml_transpose(A->matrix);

    bml_deallocate_distributed2d(B);
}
