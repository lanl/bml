#include "bml.h"
#include "bml_test.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>

int
test_function(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;
    bml_matrix_t *C = NULL;
    bml_matrix_t *D = NULL;
    bml_matrix_t *E = NULL;
    bml_matrix_t *F = NULL;

    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;
    REAL_T *C_dense = NULL;
    REAL_T *D_dense = NULL;
    REAL_T *E_dense = NULL;
    REAL_T *F_dense = NULL;

    double alpha_factor = 1.0;
    double beta_factor = 1.0;
    double threshold = 0.0;

    A = bml_random_matrix(matrix_type, matrix_precision, N, M);
    B = bml_zero_matrix(matrix_type, matrix_precision, N, M);
    C = bml_random_matrix(matrix_type, matrix_precision, N, M);
    D = bml_zero_matrix(matrix_type, matrix_precision, N, M);
    E = bml_copy_new(C);
    F = bml_zero_matrix(matrix_type, matrix_precision, N, M);

    bml_multiply(A, A, C, alpha_factor, beta_factor, threshold);
    bml_multiply_x2(A, B, threshold);

    bml_copy(A, D);
    bml_multiply(A, D, E, alpha_factor, beta_factor, threshold);
    bml_multiply_AB(A, D, F, threshold);

    A_dense = bml_convert_to_dense(A);
    B_dense = bml_convert_to_dense(B);
    C_dense = bml_convert_to_dense(C);
    D_dense = bml_convert_to_dense(D);
    E_dense = bml_convert_to_dense(E);
    F_dense = bml_convert_to_dense(F);

    bml_print_dense_matrix(N, matrix_precision, A_dense, 0, N, 0, N);
    bml_print_dense_matrix(N, matrix_precision, B_dense, 0, N, 0, N);
    bml_print_dense_matrix(N, matrix_precision, C_dense, 0, N, 0, N);
    bml_print_dense_matrix(N, matrix_precision, D_dense, 0, N, 0, N);
    bml_print_dense_matrix(N, matrix_precision, E_dense, 0, N, 0, N);
    bml_print_dense_matrix(N, matrix_precision, F_dense, 0, N, 0, N);

    for (int i = 0; i < N * N; i++)
    {
        if (fabs(C_dense[i] - E_dense[i]) > 1e-12)
        {
            LOG_ERROR
                ("matrices are not identical; C[%d] = %e E[%d] = %e\n",
                 i, C_dense[i], i, E_dense[i]);
            return -1;
        }
    }

    for (int i = 0; i < N * N; i++)
    {
        if (fabs(B_dense[i] - F_dense[i]) > 1e-12) 
        {
            LOG_ERROR
                ("matrices are not identical; B[%d] = %e F[%d] = %e\n",
                 i, B_dense[i], i, F_dense[i]);
            return -1;
        }
    }

    bml_free_memory(A_dense);
    bml_free_memory(B_dense);
    bml_free_memory(C_dense);
    bml_free_memory(D_dense);
    bml_free_memory(E_dense);
    bml_free_memory(F_dense);
    bml_deallocate(&A);
    bml_deallocate(&B);
    bml_deallocate(&C);
    bml_deallocate(&D);
    bml_deallocate(&E);
    bml_deallocate(&F);

    LOG_INFO("multiply matrix test passed\n");

    return 0;
}
