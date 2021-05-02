#include "../../macros.h"
#include "../../typed.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "../bml_allocate.h"
#include "bml_parallel_csr.h"
#include "bml_types_csr.h"
#include "bml_allocate_csr.h"
#include "../bml_logger.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

#ifdef DO_MPI
#include <mpi.h>
#endif

#ifdef DO_MPI
void TYPED_FUNC(
    bml_mpi_send_csr) (
    bml_matrix_csr_t * A,
    const int dst,
    MPI_Comm comm)
{
    int mpiret;

    // send nnz
    int *nnz = bml_allocate_memory(sizeof(int) * A->N_);
    int totnnz = 0;
    for (int i = 0; i < A->N_; i++)
    {
        nnz[i] = A->data_[i]->NNZ_;
        totnnz += nnz[i];
    }
    mpiret = MPI_Send(nnz, A->N_, MPI_INT, dst, 111, comm);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Send failed for nnz");
    bml_free_memory(nnz);

    // send cols
    int *cols = bml_allocate_memory(sizeof(int) * totnnz);
    int *pcols = cols;
    for (int i = 0; i < A->N_; i++)
    {
        csr_sparse_row_t *row = A->data_[i];
        memcpy(pcols, row->cols_, row->NNZ_ * sizeof(int));
        pcols += row->NNZ_;
    }
    mpiret = MPI_Send(cols, totnnz, MPI_INT, dst, 112, comm);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Send failed for cols");
    bml_free_memory(cols);

    // send matrix elements
    LOG_DEBUG("Send values...\n");
    REAL_T *values = bml_allocate_memory(sizeof(REAL_T) * totnnz);
    REAL_T *pvalues = values;
    for (int i = 0; i < A->N_; i++)
    {
        csr_sparse_row_t *row = A->data_[i];
        memcpy(pvalues, row->vals_, row->NNZ_ * sizeof(REAL_T));
        pvalues += row->NNZ_;
    }
    mpiret = MPI_Send(values, totnnz, MPI_T, dst, 113, comm);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Send failed for values");
    bml_free_memory(values);

    LOG_DEBUG("Done with Send...\n");
}

void TYPED_FUNC(
    bml_mpi_recv_csr) (
    bml_matrix_csr_t * A,
    const int src,
    MPI_Comm comm)
{
    MPI_Status status;
    int mpiret;

    // recv nnz
    int *nnz = bml_allocate_memory(sizeof(int) * A->N_);
    mpiret = MPI_Recv(nnz, A->N_, MPI_INT, src, 111, comm, &status);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Recv failed for nnz");
    int totnnz = 0;
    for (int i = 0; i < A->N_; i++)
    {
        A->data_[i]->NNZ_ = nnz[i];
        totnnz += nnz[i];
    }
    bml_free_memory(nnz);

    // recv columns indexes
    int *cols = bml_allocate_memory(sizeof(int) * totnnz);
    mpiret = MPI_Recv(cols, totnnz, MPI_INT, src, 112, comm, &status);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Recv failed for cols");
    int *pcols = cols;
    for (int i = 0; i < A->N_; i++)
    {
        csr_sparse_row_t *row = A->data_[i];
        memcpy(row->cols_, pcols, row->NNZ_ * sizeof(int));
        pcols += row->NNZ_;
    }
    bml_free_memory(cols);

    // recv matrix elements
    LOG_DEBUG("Recv values...\n");
    REAL_T *values = bml_allocate_memory(sizeof(REAL_T) * totnnz);
    mpiret = MPI_Recv(values, totnnz, MPI_T, src, 113, comm, &status);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Recv failed for values");
    REAL_T *pvalues = values;
    for (int i = 0; i < A->N_; i++)
    {
        csr_sparse_row_t *row = A->data_[i];
        memcpy(row->vals_, pvalues, row->NNZ_ * sizeof(REAL_T));
        pvalues += row->NNZ_;
    }
    bml_free_memory(values);

    LOG_DEBUG("Done with Recv...\n");
}

/*
 * Return BML matrix from data received from MPI task src
 */
bml_matrix_csr_t
    * TYPED_FUNC(bml_mpi_recv_matrix_csr) (int N, int M,
                                           const int src, MPI_Comm comm)
{
    bml_matrix_csr_t *A_bml =
        TYPED_FUNC(bml_zero_matrix_csr) (N, M, sequential);

    bml_mpi_recv_csr(A_bml, src, comm);

    return A_bml;
}

#endif
