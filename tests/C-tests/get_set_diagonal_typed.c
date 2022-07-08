#include "bml.h"
#include "../typed.h"
#include "bml_getters.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

int TYPED_FUNC(
    test_get_set_diagonal) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    REAL_T *A_diagonal = NULL;
    REAL_T *B_diagonal = NULL;
    double tol = 1.e-12;

    // Create a diagonal
    switch (matrix_precision)
    {
        case single_real:
            A_diagonal = calloc(N, sizeof(float));
            break;
        case double_real:
            A_diagonal = calloc(N, sizeof(double));
            break;
#ifdef BML_COMPLEX
        case single_complex:
            A_diagonal = calloc(N, sizeof(float complex));
            break;
        case double_complex:
            A_diagonal = calloc(N, sizeof(double complex));
            break;
#endif
        default:
            LOG_DEBUG("matrix_precision is not set");
            break;
    }


    for (int i = 0; i < N; i++)
    {
        A_diagonal[i] = i;
    }

    A = bml_random_matrix(matrix_type, matrix_precision, N, M, sequential);
    REAL_T *A_dense = bml_export_to_dense(A, dense_row_major);

    bml_set_diagonal(A, A_diagonal, 0.01);

    LOG_INFO("Random matrix with set diagonal\n");
    bml_print_bml_matrix(A, 0, N, 0, N);

    B_diagonal = bml_get_diagonal(A);

    for (int i = 0; i < N; i++)
    {
        if (ABS(A_diagonal[i] - B_diagonal[i]) > tol)
        {
            LOG_ERROR
                ("bml_get_diagonal and/or bml_set_diagonal are corrupted\n");
            return -1;
        }
    }

    // check no off-diagonal elements was reset
    REAL_T *B_dense = bml_export_to_dense(A, dense_row_major);
    for (int i = 0; i < N * N; i++)
    {
        if (ABS(A_dense[i] - B_dense[i]) > tol && (i % (N + 1) != 0))
        {
            LOG_ERROR("matrices not identical A[%d] = %e, B[%d] = %e\n",
                      i, A_dense[i], i, B_dense[i]);
            return -1;
        }
    }

    free(B_diagonal);
    bml_free_memory(A_dense);
    bml_free_memory(B_dense);
    bml_deallocate(&A);
    LOG_INFO("Test set/get diagonal for random matrix passed\n");

    // 2nd test: set diagonal of zero matrix with two successive calls
    A = bml_zero_matrix(matrix_type, matrix_precision, N, M, sequential);

    // set diagonal with large threshold
    // (should be setting a subset of diagonal entries)
    bml_set_diagonal(A, A_diagonal, 4.);

    // set diagonal again with smaller threshold
    // (should be setting more diagonal entries)
    bml_set_diagonal(A, A_diagonal, 0.01);

    LOG_INFO("Zero matrix with set diagonal\n");
    bml_print_bml_matrix(A, 0, N, 0, N);

    B_diagonal = bml_get_diagonal(A);

    for (int i = 0; i < N; i++)
    {
        if (ABS(A_diagonal[i] - B_diagonal[i]) > tol)
        {
            LOG_ERROR
                ("bml_get_diagonal and/or bml_set_diagonal are corrupted\n");
            return -1;
        }
    }

    free(A_diagonal);
    free(B_diagonal);
    bml_deallocate(&A);

    LOG_INFO("Test set/get diagonal for zero matrix passed\n");

    return 0;
}
