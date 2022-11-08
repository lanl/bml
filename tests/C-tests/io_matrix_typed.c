#define _XOPEN_SOURCE 500

#include "bml.h"
#include "bml_test.h"

#include "../typed.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int TYPED_FUNC(
    test_io_matrix) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;

    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef BML_USE_MPI
    if (bml_getNRanks() > 1)
    {
        LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif

    double diff;
    double tol;

    char *matrix_filename;
    if (bml_getMyRank() == 0)
    {
        matrix_filename = strdup("ctest_matrix_XXXXXX");
        mktemp(matrix_filename);
        LOG_INFO("Using %s as matrix file\n", matrix_filename);
    }

    A = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);
    bml_write_bml_matrix(A, matrix_filename);
    B = bml_zero_matrix(matrix_type, matrix_precision, N, M, distrib_mode);
    bml_read_bml_matrix(B, matrix_filename);

    A_dense = bml_export_to_dense(A, dense_row_major);
    B_dense = bml_export_to_dense(B, dense_row_major);

    if (bml_getMyRank() == 0)
    {
        LOG_INFO("A (random matrix):\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense,
                               0, N, 0, N);
        LOG_INFO("B (matrix read from file)):\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, B_dense,
                               0, N, 0, N);
    }

    if (matrix_precision == single_real || matrix_precision == single_complex)
    {
        tol = 1e-6;
    }
    else
    {
        tol = 1e-12;
    }

    if (bml_getMyRank() == 0)
    {
        for (int i = 0; i < N * N; i++)
        {
            diff = ABS(A_dense[i] - B_dense[i]);
            if (diff > tol)
            {
                LOG_ERROR
                    ("matrices are not identical; A[%d] = %e, B[%d] = %e, diff = %e\n",
                     i, A_dense[i], i, B_dense[i], diff);
                return -1;
            }
        }
        free(matrix_filename);
        bml_free_memory(A_dense);
        bml_free_memory(B_dense);
    }
    bml_deallocate(&A);
    bml_deallocate(&B);
    if (bml_getMyRank() == 0)
        LOG_INFO("io matrix test passed\n");
    return 0;
}
