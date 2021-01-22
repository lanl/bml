#include "bml.h"
#include "../typed.h"
#include "bml_parallel.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>

#ifdef DO_MPI

int TYPED_FUNC(
    test_mpi_sendrecv) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A =
        bml_random_matrix(matrix_type, matrix_precision, N, M, sequential);
    REAL_T *A_dense = bml_export_to_dense(A, dense_row_major);

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
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

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (i == j)
                {
                    if (ABS(A_dense[i * N + j] - B_dense[i * N + j]) > 1e-12)
                    {
                        LOG_ERROR
                            ("incorrect value on diagonal; A[%d,%d] = %e B[%d,%d] = %e\n",
                             i, i, A_dense[i * N + j], i, i,
                             B_dense[i * N + j]);
                        return -1;
                    }
                }
                else
                {
                    if (ABS(A_dense[i * N + j] - B_dense[i * N + j]) > 1e-12)
                    {
                        LOG_ERROR
                            ("incorrect value off-diagonal; A[%d,%d] = %e B[%d,%d] = %e\n",
                             i, j, A_dense[i * N + j], i, i,
                             B_dense[i * N + j]);
                        return -1;
                    }
                }
            }
        }

        LOG_INFO("send/recv test passed\n");
        bml_free_memory(B_dense);
        bml_deallocate(&B);
    }

    bml_clear(A);
    bml_deallocate(&A);
    bml_free_memory(A_dense);

    return 0;
}

#endif
