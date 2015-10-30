#include "bml.h"
#include "bml_test.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_TOL 1e-6

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

    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;
    REAL_T *C_dense = NULL;

    double traceA, traceB, traceC;
    double scalar = 0.8;

    double rel_diff;

    A = bml_identity_matrix(matrix_type, matrix_precision, N, M);
    B = bml_scale_new(scalar, A);
    C = bml_scale_new(scalar, B);

    traceA = bml_trace(A);
    traceB = bml_trace(B);
    traceC = bml_trace(C);

    A_dense = bml_convert_to_dense(A);
    B_dense = bml_convert_to_dense(B);
    C_dense = bml_convert_to_dense(C);

    bml_print_dense_matrix(N, matrix_precision, A_dense, 0, N, 0, N);
    bml_print_dense_matrix(N, matrix_precision, B_dense, 0, N, 0, N);
    bml_print_dense_matrix(N, matrix_precision, C_dense, 0, N, 0, N);

    printf("traceA = %e (%e), diff. traceA = %e\n", traceA, (double) N,
           traceA - (double) N);
    printf("traceB = %e (%e), diff. traceB = %e\n", traceB, scalar * N,
           traceB - scalar * N);
    printf("traceC = %e (%e), diff. traceC = %e\n", traceC,
           scalar * scalar * N, traceC - scalar * scalar * N);

    if ((rel_diff = fabs(traceA - (double) N) / (double) N) > REL_TOL)
    {
        LOG_ERROR
            ("traces are not correct; traceA = %e and not %e, rel.diff = %e\n",
             traceA, rel_diff);
        return -1;
    }
    if ((rel_diff =
         fabs(traceB - (double) (scalar * N)) / (double) (scalar * N)) >
        REL_TOL)
    {
        LOG_ERROR
            ("traces are not correct; traceB = %e and not %e, rel.diff = %e\n",
             traceB, (double) (scalar * N), rel_diff);
        return -1;
    }
    if ((rel_diff =
         fabs(traceC -
              (double) (scalar * scalar * N)) / (double) (scalar * N * N)) >
        REL_TOL)
    {
        LOG_ERROR
            ("traces are not correct; traceC = %e and not %e, rel.diff = %e\n",
             traceC, (double) (scalar * scalar * N), rel_diff);
        return -1;
    }
    bml_free_memory(A_dense);
    bml_free_memory(B_dense);
    bml_free_memory(C_dense);
    bml_deallocate(&A);
    bml_deallocate(&B);
    bml_deallocate(&C);

    LOG_INFO("trace matrix test passed\n");

    return 0;
}
