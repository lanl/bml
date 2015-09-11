#include "bml.h"
#include "bml_test.h"

#include <math.h>
#include <stdlib.h>

int test_function(const int N,
                  const bml_matrix_type_t matrix_type,
                  const bml_matrix_precision_t matrix_precision)
{
    bml_matrix_t *A;
    float *A_float;
    float *B_float;
    double *A_double;
    double *B_double;

    switch(matrix_precision) {
    case single_precision:
        A_float = bml_allocate_memory(sizeof(float)*N*N);
        for(int i = 0; i < N*N; i++) {
            A_float[i] = rand()/(float) RAND_MAX;
        }
        bml_print_matrix(N, 0, N, 0, N, matrix_precision, A_float);
        A = bml_convert_from_dense(matrix_type, matrix_precision, N, A_float, 0);
        B_float = bml_convert_to_dense(A);
        for(int i = 0; i < N*N; i++) {
            if(fabs(A_float[i]-B_float[i]) > 1e-12) {
                LOG_ERROR("matrix element mismatch\n");
            }
        }
        bml_free_memory(A_float);
        bml_free_memory(B_float);
        break;
    case double_precision:
        A_double = bml_allocate_memory(sizeof(double)*N*N);
        for(int i = 0; i < N*N; i++) {
            A_double[i] = rand()/(double) RAND_MAX;
        }
        bml_print_matrix(N, 0, N, 0, N, matrix_precision, A_double);
        A = bml_convert_from_dense(matrix_type, matrix_precision, N, A_double, 0);
        B_double = bml_convert_to_dense(A);
        for(int i = 0; i < N*N; i++) {
            if(fabs(A_double[i]-B_double[i]) > 1e-12) {
                LOG_ERROR("matrix element mismatch\n");
            }
        }
        bml_free_memory(A_double);
        bml_free_memory(B_double);
        break;
    }
    bml_deallocate(&A);
    return 0;
}
