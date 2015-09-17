#include "bml.h"
#include "bml_test.h"

#include <math.h>
#include <stdlib.h>

int test_function(const int N,
                  const bml_matrix_type_t matrix_type,
                  const bml_matrix_precision_t matrix_precision,
                  const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;

    float *A_float = NULL, *B_float = NULL;
    double *A_double = NULL, *B_double = NULL;

    A = bml_random_matrix(matrix_type, matrix_precision, N, M);
    B = bml_copy(A);

    switch(matrix_precision) {
    case single_real:
        A_float = bml_convert_to_dense(A);
        B_float = bml_convert_to_dense(B);
        bml_print_matrix(N, matrix_precision, A_float, 0, N, 0, N);
        bml_print_matrix(N, matrix_precision, B_float, 0, N, 0, N);
       for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(fabs(A_float[i+j*N] - B_float[i+j*N]) > 1e-12) {
                    LOG_ERROR("matrices are not identical; A[%d,%d] = %e\n", i, j, A_float[i*N+j]);
                    return -1;
                }
            }
        }
        bml_free_memory(A_float);
        bml_free_memory(B_float);
        break;
    case double_real:
        A_double = bml_convert_to_dense(A);
        B_double = bml_convert_to_dense(B);
        bml_print_matrix(N, matrix_precision, A_double, 0, N, 0, N);
        bml_print_matrix(N, matrix_precision, B_double, 0, N, 0, N);
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(fabs(A_double[i+j*N] - B_double[i+j*N]) > 1e-12) {
                    LOG_ERROR("matrices are not identical; A[%d,%d] = %e\n", i, j, A_double[i*N+j]);
                    return -1;
                }
            }
        }
        bml_free_memory(A_double);
        bml_free_memory(B_double);
        break;
    }
    LOG_INFO("identity matrix test passed\n");

    return 0;
}
