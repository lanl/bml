#include "bml.h"
#include "../typed.h"
#include "../macros.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
#define REL_TOL 1e-6
#else
#define REL_TOL 1e-12
#endif

int TYPED_FUNC(
    test_add) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;
    bml_matrix_t *C = NULL;

    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;
    REAL_T *C_dense = NULL;

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef DO_MPI
    if (bml_getNRanks() > 1)
    {
        LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif

    double alpha = 1.2;
    double beta = 0.8;
    double threshold = 0.0;

    int max_row = MIN(N, PRINT_THRESHOLD);
    int max_col = MIN(N, PRINT_THRESHOLD);

    LOG_DEBUG("rel. tolerance = %e\n", REL_TOL);

    A = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);
    B = bml_copy_new(A);
    C = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);

    LOG_DEBUG("bml_add...\n");
    bml_add(B, C, alpha, beta, threshold);

    LOG_INFO("alpha = %f \n ", alpha);
    LOG_INFO("beta = %f \n ", beta);
    A_dense = bml_export_to_dense(A, dense_row_major);
    B_dense = bml_export_to_dense(B, dense_row_major);
    C_dense = bml_export_to_dense(C, dense_row_major);

    if (bml_getMyRank() == 0)
    {
        LOG_INFO("A\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense,
                               0, max_row, 0, max_col);
        LOG_INFO("C\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, C_dense,
                               0, max_row, 0, max_col);
        LOG_INFO("B = alpha A + beta C \n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, B_dense,
                               0, max_row, 0, max_col);

        for (int i = 0; i < N * N; i++)
        {
            double expected = alpha * A_dense[i] + beta * C_dense[i];
            double rel_diff_val = (expected - B_dense[i]) / expected;
            double rel_diff = fabs(rel_diff_val);
            if (rel_diff > REL_TOL)
            {
                LOG_ERROR
                    ("matrices are not identical; expected[%d] = %e, B[%d] = %e\n",
                     i, expected, i, B_dense[i]);
                return -1;
            }
        }
        bml_free_memory(A_dense);
        bml_free_memory(B_dense);
        bml_free_memory(C_dense);
    }

    bml_deallocate(&A);
    bml_deallocate(&B);
    bml_deallocate(&C);

    LOG_INFO("add matrix test passed\n");

    return 0;
}
