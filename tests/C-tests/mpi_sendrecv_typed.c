#include "bml.h"
#include "../typed.h"
#include "bml_parallel.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>

#ifdef DO_MPI

static int TYPED_FUNC(
    compare_matrices) (
    const int N,
    const REAL_T * A_dense,
    const REAL_T * B_dense,
    const double tol)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (ABS(A_dense[i * N + j] - B_dense[i * N + j]) > tol)
            {
                LOG_ERROR
                    ("incorrect values: A[%d,%d] = %e, B[%d,%d] = %e\n",
                     i, j, A_dense[i * N + j], i, j, B_dense[i * N + j]);
                return -1;
            }
        }
    }
    return 0;
}

int TYPED_FUNC(
    test_mpi_sendrecv) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    bml_matrix_t *A =
        bml_random_matrix(matrix_type, matrix_precision, N, M, sequential);
    if (myrank == 0)
        bml_print_bml_matrix(A, 0, N, 0, N);

    // test point-to-point communications
    {
        REAL_T *A_dense = bml_export_to_dense(A, dense_row_major);

        if (myrank % 2 == 1)
        {
            LOG_DEBUG("Send matrix\n");
            bml_mpi_send(A, myrank - 1, MPI_COMM_WORLD);
            MPI_Send(A_dense, N * M, MPI_T, myrank - 1, 4567, MPI_COMM_WORLD);
        }
        else if (myrank % 2 == 0)
        {
            LOG_DEBUG("Recv matrix\n");
            // receive copy of A into B
            bml_matrix_t *B =
                bml_mpi_recv_matrix(matrix_type, matrix_precision, N, M,
                                    myrank + 1, MPI_COMM_WORLD);

            MPI_Status status;
            MPI_Recv(A_dense, N * M, MPI_T, myrank + 1, 4567, MPI_COMM_WORLD,
                     &status);

            LOG_DEBUG("Compare matrix\n");
            // compare A and B
            REAL_T *B_dense = bml_export_to_dense(B, dense_row_major);

//        bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense,
//                               0, N, 0, N);
//        bml_print_dense_matrix(N, matrix_precision, dense_row_major, B_dense,
//                               0, N, 0, N);

            int ret =
                TYPED_FUNC(compare_matrices) (N, A_dense, B_dense, 1.e-12);
            if (ret != 0)
                return ret;

            LOG_INFO("send/recv test passed\n");
            bml_free_memory(B_dense);
            bml_deallocate(&B);
        }
        if (myrank == 0)
            bml_free_memory(A_dense);
    }

    // test collective communications
    {
        REAL_T *A_dense = bml_export_to_dense(A, dense_row_major);
        // bcast a BML matrix
        bml_mpi_bcast_matrix(A, 0, MPI_COMM_WORLD);

        if (myrank != 0)
            A_dense = bml_allocate_memory(sizeof(REAL_T) * N * N);
        MPI_Bcast(A_dense, N * N, MPI_T, 0, MPI_COMM_WORLD);
        REAL_T *B_dense = bml_export_to_dense(A, dense_row_major);

        int ret = TYPED_FUNC(compare_matrices) (N, A_dense, B_dense, 1.e-12);
        if (ret != 0)
            return ret;

        LOG_INFO("bcast test passed\n");
        bml_free_memory(A_dense);
    }

    bml_clear(A);
    bml_deallocate(&A);

    return 0;
}

#endif
