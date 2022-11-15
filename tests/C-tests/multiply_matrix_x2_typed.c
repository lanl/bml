#include "bml.h"
#include "../typed.h"
#include "../macros.h"
#include "ellblock/bml_allocate_ellblock.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static void TYPED_FUNC(
    ref_multiply) (
    const int N,
    const REAL_T * A,
    const REAL_T * B,
    REAL_T * C,
    const double alpha,
    const double beta,
    const double threshold)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] *= beta;
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] += alpha * A[i * N + k] * B[k * N + j];
            }
            REAL_T val = C[i * N + j];
            C[i * N + j] = ABS(val) > threshold ? val : 0.;
        }
    }
}

#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
#define ABS_TOL 2e-6f
#else
#define ABS_TOL 1e-12
#endif

static int TYPED_FUNC(
    compare_matrix) (
    const int N,
    const bml_matrix_precision_t matrix_precision,
    REAL_T * A,
    REAL_T * B)
{
    int max_row = MIN(N, PRINT_THRESHOLD);
    int max_col = MIN(N, PRINT_THRESHOLD);

    for (int i = 0; i < N * N; i++)
    {
        if (ABS(A[i] - B[i]) > ABS_TOL)
        {
            bml_print_dense_matrix(N, matrix_precision, dense_row_major, A, 0,
                                   max_row, 0, max_col);
            bml_print_dense_matrix(N, matrix_precision, dense_row_major, B, 0,
                                   max_row, 0, max_col);
            LOG_INFO("element %d outside %1.2e\n", i, ABS_TOL);
#if defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX)
            LOG_INFO("A[%d] = %e+%ei\n", i, creal(A[i]), cimag(A[i]));
            LOG_INFO("B[%d] = %e+%ei\n", i, creal(B[i]), cimag(B[i]));
            LOG_INFO("abs(A[%d]-B[%d]) = %e\n", i, i, ABS(A[i] - B[i]));
#else
            LOG_INFO("A[%d] = %e\n", i, A[i]);
            LOG_INFO("B[%d] = %e\n", i, B[i]);
            LOG_INFO("abs(A[%d]-B[%d]) = %e\n", i, i, ABS(A[i] - B[i]));
#endif
            return 1;
        }
    }
    return 0;
}

int TYPED_FUNC(
    test_multiply_x2) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    // set block sizes for ellblock tests
    // (unused by other tests)
    if (matrix_type == ellblock)
    {
        int bsizes[N];
        int count = 0;
        int nb = 0;
        for (int i = 0; i < N; i++)
        {
            int bsize = (5 + i) % 6 + 1;
            count += bsize;
            if (count > N)
            {
                bsize -= (count - N);
                count = N;
            }
            bsizes[i] = bsize;
            if (count == N)
            {
                nb = i + 1;
                break;
            }
        }
        bml_set_block_sizes(bsizes, nb, nb * M / N);
        for (int i = 0; i < nb; i++)
        {
            printf("bsizes[%d] = %d\n", i, bsizes[i]);
        }
    }
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;
    bml_matrix_t *C = NULL;

    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;
    REAL_T *C_dense = NULL;
    REAL_T *D_dense = NULL;
    REAL_T *E_dense = NULL;

    const double alpha = 1.0;
    const double beta = 0.0;
    const double threshold = 0.0;
    REAL_T *trace;


    A = bml_random_matrix(matrix_type, matrix_precision, N, M, sequential);
    B = bml_copy_new(A);
    C = bml_random_matrix(matrix_type, matrix_precision, N, M, sequential);

    int max_row = MIN(N, PRINT_THRESHOLD);
    int max_col = MIN(N, PRINT_THRESHOLD);

    LOG_INFO("A\n");
    bml_print_bml_matrix(A, 0, max_row, 0, max_col);
    LOG_INFO("B = A\n");
    bml_print_bml_matrix(B, 0, max_row, 0, max_col);
    LOG_INFO("C\n");
    bml_print_bml_matrix(C, 0, max_row, 0, max_col);

    A_dense = bml_export_to_dense(A, dense_row_major);
    B_dense = bml_export_to_dense(B, dense_row_major);
    C_dense = bml_export_to_dense(C, dense_row_major);
    D_dense = bml_export_to_dense(C, dense_row_major);

    //bml_multiply(A, B, C, alpha, beta, threshold);
    trace = bml_multiply_x2(A, C, threshold);

    LOG_INFO("A^2\n");
    bml_print_bml_matrix(C, 0, max_row, 0, max_col);

    E_dense = bml_export_to_dense(C, dense_row_major);

    TYPED_FUNC(ref_multiply) (N, A_dense, B_dense, D_dense, alpha, beta,
                              threshold);

    if (TYPED_FUNC(compare_matrix) (N, matrix_precision, D_dense, E_dense) !=
        0)
    {
        LOG_ERROR("matrix product incorrect\n");
        return -1;
    }
    LOG_INFO("multiply matrix test passed\n");

    bml_deallocate(&A);
    bml_deallocate(&B);
    bml_deallocate(&C);

    bml_free_memory(A_dense);
    bml_free_memory(B_dense);
    bml_free_memory(C_dense);
    bml_free_memory(D_dense);
    bml_free_memory(E_dense);
    bml_free_memory(trace);

    return 0;
}
