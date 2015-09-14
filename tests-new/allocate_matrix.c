#include "bml.h"
#include "bml_test.h"

#include <math.h>
#include <stdlib.h>

int test_function(const int N,
                  const bml_matrix_type_t matrix_type,
                  const bml_matrix_precision_t matrix_precision)
{
    bml_matrix_t *A = NULL;

    float *A_float = NULL;
    double *A_double = NULL;

    A = bml_random_matrix(matrix_type, matrix_precision, N);
    bml_deallocate(&A);
    A = bml_identity_matrix(matrix_type, matrix_precision, N);
    switch(matrix_precision) {
    case single_precision:
        A_float = bml_convert_to_dense(A);
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(i == j) {
                    if(fabs(A_float[i+j*N]-1) > 1e-12) {
                        LOG_ERROR("incorrect value on diagonal; A[%d,%d] = %e\n", i, i, A_float[i+j*N]);
                        return -1;
                    }
                } else {
                    if(fabs(A_float[i+j*N]) > 1e-12) {
                        LOG_ERROR("incorrect value off-diagonal; A[%d,%d] = %e\n", i, j, A_float[i+j*N]);
                        return -1;
                    }
                }
            }
        }
        bml_free_memory(A_float);
        break;
    case double_precision:
        A_double = bml_convert_to_dense(A);
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(i == j) {
                    if(fabs(A_double[i+j*N]-1) > 1e-12) {
                        LOG_ERROR("incorrect value on diagonal; A[%d,%d] = %e\n", i, i, A_double[i+j*N]);
                        return -1;
                    }
                } else {
                    if(fabs(A_double[i+j*N]) > 1e-12) {
                        LOG_ERROR("incorrect value off-diagonal; A[%d,%d] = %e\n", i, j, A_double[i+j*N]);
                        return -1;
                    }
                }
            }
        }
        bml_free_memory(A_double);
        break;
    }
    LOG_INFO("identity matrix test passed\n");

    bml_deallocate(&A);
    A = bml_zero_matrix(matrix_type, matrix_precision, N);
    bml_deallocate(&A);

    return 0;
}
