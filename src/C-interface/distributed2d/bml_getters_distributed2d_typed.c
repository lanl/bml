#include "../../macros.h"
#include "../../typed.h"
#include "../bml_logger.h"
#include "../bml_setters.h"
#include "../bml_getters.h"
#include "../bml_allocate.h"
#include "bml_getters_distributed2d.h"

void *TYPED_FUNC(
    bml_get_row_distributed2d) (
    bml_matrix_distributed2d_t * A,
    int i)
{
    const int nloc = A->N / A->nprows;
    // allocate full row to be returned
    REAL_T *row = bml_allocate_memory(A->N * sizeof(REAL_T));

    REAL_T *sub_row = NULL;
    int *counts = calloc(A->ntasks, sizeof(int));

    int mycount = 0;
    // get (piece of) row if "i" corresponds to local row
    if ((i >= A->myprow * nloc) && i < (A->myprow + 1) * nloc)
    {
        int irow = i - A->myprow * nloc;
        sub_row = bml_get_row(A->matrix, irow);
        mycount = nloc;
    }

    int *recvcounts = calloc(A->ntasks, sizeof(int));
    int ret =
        MPI_Allgather(&mycount, 1, MPI_INT, recvcounts, 1, MPI_INT, A->comm);
    if (ret == MPI_ERR_COUNT)
        LOG_ERROR("MPI_ERR_COUNT");
    if (ret == MPI_ERR_BUFFER)
        LOG_ERROR("MPI_ERR_BUFFER");

    int *displs = calloc(A->ntasks, sizeof(int));
    for (int i = 0; i < A->ntasks - 1; i++)
        displs[i + 1] += recvcounts[i];

    MPI_Allgatherv(sub_row, mycount, MPI_T, row, recvcounts, displs, MPI_T,
                   A->comm);
    if (ret == MPI_ERR_COUNT)
        LOG_ERROR("bml_get_row_distributed2d: MPI_Allgatherv, MPI_ERR_COUNT");
    if (ret == MPI_ERR_BUFFER)
        LOG_ERROR
            ("bml_get_row_distributed2d: MPI_Allgatherv, MPI_ERR_BUFFER");

    free(recvcounts);
    free(counts);
    free(displs);

    return row;
}
