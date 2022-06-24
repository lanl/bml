// This code runs bml_multiply and bml_multiply_x2.
// This template will be used for a roofline analysis.

#include "bml.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>

int
main(
    )
{
    bml_matrix_t *A;
    bml_matrix_t *B;
    bml_matrix_t *C;

    const int N = 1000;
    const int M = 1000;
    bml_matrix_precision_t matrix_precision = double_real;
    bml_matrix_type_t matrix_type = dense;
    const double alpha = 1.2;
    const double beta = 0.8;
    const double threshold = 0.0;
    double *trace;

    A = bml_random_matrix(matrix_type, matrix_precision, N, M, sequential);
    B = bml_random_matrix(matrix_type, matrix_precision, N, M, sequential);
    C = bml_random_matrix(matrix_type, matrix_precision, N, M, sequential);

    trace = bml_multiply_x2(A, C, threshold);   // C=A*A
    bml_multiply(A, B, C, alpha, beta, threshold);      // C=alpha* A*B + beta*C

    free(trace);
    bml_deallocate(&A);
    bml_deallocate(&B);
    bml_deallocate(&C);

    return 0;
}
