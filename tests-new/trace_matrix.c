#include "bml.h"
#include "bml_test.h"

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

    float *A_float = NULL, *B_float = NULL, *C_float = NULL;
    double *A_double = NULL, *B_double = NULL, *C_double = NULL;

    double traceA, traceB, traceC;
    double scalar = 5.0;

    A = bml_identity_matrix(matrix_type, matrix_precision, N, M);
    traceA = bml_trace(A);
    B = bml_scale_new(scalar, A);
    traceB = bml_trace(B);
    C = bml_scale_new(scalar, B);
    traceC = bml_trace(C);

    switch (matrix_precision)
    {
    case single_real:
        A_float = bml_convert_to_dense(A);
        B_float = bml_convert_to_dense(B);
        C_float = bml_convert_to_dense(C);
        bml_print_dense_matrix(N, matrix_precision, A_float, 0, N, 0, N);
        bml_print_dense_matrix(N, matrix_precision, B_float, 0, N, 0, N);
        bml_print_dense_matrix(N, matrix_precision, C_float, 0, N, 0, N);
    
        if (fabs(traceA - (double)(N)) > 1e-12 ||
            fabs(traceB - (double)(scalar * N)) > 1e-12 || 
            fabs(traceC - (double)(scalar * scalar * N)) > 1e-12) 
        {
                LOG_ERROR("traces are not correct; traceA = %e traceB = %e traceC = %e\n", traceA, traceB, traceC);
                return -1;
        }
        bml_free_memory(A_float);
        bml_free_memory(B_float);
        bml_free_memory(C_float);
        break;
    case double_real:
        A_double = bml_convert_to_dense(A);
        B_double = bml_convert_to_dense(B);
        C_double = bml_convert_to_dense(C);;
        bml_print_dense_matrix(N, matrix_precision, A_double, 0, N, 0, N);
        bml_print_dense_matrix(N, matrix_precision, B_double, 0, N, 0, N);
        bml_print_dense_matrix(N, matrix_precision, C_double, 0, N, 0, N);
 
        if (fabs(traceA - (double)(N)) > 1e-12 ||
            fabs(traceB - (double)(scalar * N)) > 1e-12 ||
            fabs(traceC - (double)(scalar * scalar * N)) > 1e-12)
        {
                LOG_ERROR("traces are not correct; traceA = %e traceB = %e traceC = %e\n", traceA, traceB, traceC);
                return -1;
        }
        bml_free_memory(A_double);
        bml_free_memory(B_double);
        bml_free_memory(C_double);
        break;
    default:
        LOG_ERROR("unknown precision\n");
        return -1;
        break;
    }
    bml_deallocate(&A);
    bml_deallocate(&B);
    bml_deallocate(&C);

    LOG_INFO("trace matrix test passed\n");

    return 0;
}
