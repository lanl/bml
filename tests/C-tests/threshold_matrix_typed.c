#include "bml.h"
#include "../typed.h"
#include "../macros.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

int TYPED_FUNC(
    test_threshold) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    int max_row = MIN(N, PRINT_THRESHOLD);
    int max_col = MIN(N, PRINT_THRESHOLD);

    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;

    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;

    double threshold = 0.5;

#ifdef BML_USE_MAGMA
    if (matrix_type == dense)
    {
        LOG_INFO("submatrix matrix test not available\n");
        return 0;
    }
#endif

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef BML_USE_MPI
    if (bml_getNRanks() > 1)
    {
        LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif

    // test function bml_threshold_new()
    A = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);
    LOG_INFO("bml_threshold_new...\n");
    B = bml_threshold_new(A, threshold);

    A_dense = bml_export_to_dense(A, dense_row_major);
    B_dense = bml_export_to_dense(B, dense_row_major);
    if (bml_getMyRank() == 0)
    {
        printf("A \n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense,
                               0, N, 0, N);
        printf("B (A thresholded at %f \n", threshold);
        bml_print_dense_matrix(N, matrix_precision, dense_row_major,
                               B_dense, 0, N, 0, N);
        for (int i = 0; i < N * N; i++)
        {
            if (ABS(B_dense[i]) > 0 && ABS(B_dense[i]) < threshold)
            {
                LOG_ERROR("matrices not thresholded B[%d] = %e\n", i,
                          B_dense[i]);
                return -1;
            }
        }

        bml_free_memory(A_dense);
        bml_free_memory(B_dense);
    }
    bml_deallocate(&A);
    bml_deallocate(&B);

    // now test function bml_threshold()
    A = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);
    REAL_T scale_factor = 1.e-2;
    bml_scale_inplace(&scale_factor, A);
    if (bml_getMyRank() == 0)
    {
        printf("A scaled by %f \n", scale_factor);
    }
    bml_print_bml_matrix(A, 0, max_row, 0, max_col);

    REAL_T *diagonal = bml_allocate_memory(N * sizeof(REAL_T));
    for (int i = 0; i < N; i++)
        diagonal[i] = 1.;
    bml_set_diagonal(A, diagonal, 0.);
    LOG_INFO("Scaled matrix\n");
    bml_print_bml_matrix(A, 0, N, 0, N);
    bml_free_memory(diagonal);
    bml_threshold(A, 2. * scale_factor);

    B = bml_identity_matrix(matrix_type, matrix_precision, N, M,
                            distrib_mode);

    A_dense = bml_export_to_dense(A, dense_row_major);
    B_dense = bml_export_to_dense(B, dense_row_major);
    LOG_INFO("Thresholded matrix\n");
    if (bml_getMyRank() == 0)
    {
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense,
                               0, N, 0, N);

        double tol = 1.e-6;
        for (int i = 0; i < N * N; i++)
        {
            if (ABS(B_dense[i] - A_dense[i]) > tol)
            {
                LOG_ERROR("matrices not identical A[%d] = %e, B[%d] = %e\n",
                          i, A_dense[i], i, B_dense[i]);
                return -1;
            }
        }
        bml_free_memory(A_dense);
        bml_free_memory(B_dense);
    }
    bml_deallocate(&A);
    bml_deallocate(&B);

    return 0;
}
