#include "../../macros.h"
#include "../../typed.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "../bml_allocate.h"
#include "bml_parallel_csr.h"
#include "bml_types_csr.h"
#include "bml_allocate_csr.h"
#include "bml_setters_csr.h"
#include "../bml_logger.h"
#include "../bml_utilities.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef BML_USE_MPI
#include <mpi.h>
#endif

#ifdef BML_USE_MPI
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
}

void TYPED_FUNC(
    bml_mpi_irecv_csr) (
    bml_matrix_csr_t * A,
    const int src,
    MPI_Comm comm)
{
    // recv nnz for each row
    A->nnz_buffer = bml_allocate_memory(sizeof(int) * A->N_);
    int mpiret =
        MPI_Irecv(A->nnz_buffer, A->N_, MPI_INT, src, 111, comm, A->req);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Irecv failed for nnz");

    // estimate total number of non-zero one may receive
    int totnnz = A->NZMAX_ * A->N_ * 2;

    // receive column indexes
    A->cols_buffer = bml_allocate_memory(sizeof(int) * totnnz);
    mpiret =
        MPI_Irecv(A->cols_buffer, totnnz, MPI_INT, src, 112, comm,
                  A->req + 1);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Irecv failed for cols");

    // recv matrix elements
    A->buffer = bml_allocate_memory(sizeof(REAL_T) * totnnz);
    mpiret = MPI_Irecv(A->buffer, totnnz, MPI_T, src, 113, comm, A->req + 2);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Irecv failed for values");
}

void TYPED_FUNC(
    bml_mpi_irecv_complete_csr) (
    bml_matrix_csr_t * A)
{
    MPI_Waitall(3, A->req, MPI_STATUS_IGNORE);

    // move data from receive buffer into matrix
    REAL_T *pvalues = A->buffer;
    int *pcols = A->cols_buffer;
    for (int i = 0; i < A->N_; i++)
    {
        TYPED_FUNC(bml_set_sparse_row_csr) (A, i, A->nnz_buffer[i], pcols,
                                            pvalues, 0.);
        pvalues += A->nnz_buffer[i];
        pcols += A->nnz_buffer[i];
    }
    bml_free_memory(A->nnz_buffer);
    bml_free_memory(A->buffer);
    bml_free_memory(A->cols_buffer);
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

void TYPED_FUNC(
    bml_mpi_bcast_matrix_csr) (
    bml_matrix_csr_t * A,
    const int root,
    MPI_Comm comm)
{
    assert(A->N_ > 0);

    int totnnz = 0;
    if (bml_getMyRank() == root)
        for (int i = 0; i < A->N_; i++)
        {
            totnnz += A->data_[i]->NNZ_;
        }
    MPI_Bcast(&totnnz, 1, MPI_INT, root, comm);
    assert(totnnz > 0);

    // bcast cols
    int *cols = bml_allocate_memory(sizeof(int) * totnnz);
    int *pcols = cols;
    if (bml_getMyRank() == root)
        for (int i = 0; i < A->N_; i++)
        {
            csr_sparse_row_t *row = A->data_[i];
            memcpy(pcols, row->cols_, row->NNZ_ * sizeof(int));
            pcols += row->NNZ_;
        }
    MPI_Bcast(cols, totnnz, MPI_INT, root, comm);

    // bcast nnz
    int *nnz = bml_allocate_memory(sizeof(int) * A->N_);
    if (bml_getMyRank() == root)
        for (int i = 0; i < A->N_; i++)
        {
            csr_sparse_row_t *row = A->data_[i];
            nnz[i] = row->NNZ_;
        }
    MPI_Bcast(nnz, A->N_, MPI_INT, root, comm);

    // bcast matrix elements
    REAL_T *values = bml_allocate_memory(sizeof(REAL_T) * totnnz);
    REAL_T *pvalues = values;
    if (bml_getMyRank() == root)
        for (int i = 0; i < A->N_; i++)
        {
            csr_sparse_row_t *row = A->data_[i];
            memcpy(pvalues, row->vals_, row->NNZ_ * sizeof(REAL_T));
            pvalues += row->NNZ_;
        }
    MPI_Bcast(values, totnnz, MPI_T, root, comm);

    // assign received data
    pvalues = values;
    pcols = cols;
    for (int i = 0; i < A->N_; i++)
    {
        TYPED_FUNC(csr_set_sparse_row) (A->data_[i], nnz[i], pcols, pvalues,
                                        0.);
        pvalues += nnz[i];
        pcols += nnz[i];
    }

    bml_free_memory(values);
    bml_free_memory(cols);
    bml_free_memory(nnz);
}

#endif
