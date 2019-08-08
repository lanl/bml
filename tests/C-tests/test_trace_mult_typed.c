#include "bml.h"
#include "../typed.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
#define REL_TOL 1e-6
#else
#define REL_TOL 1e-12
#endif

int TYPED_FUNC(
    test_trace_mult) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;

    double trace, trace_mult, diff, threshold;
    threshold = 0.0;
    REAL_T scalar = 0.5;

    A = bml_identity_matrix(matrix_type, matrix_precision, N, M, sequential);
    B = bml_scale_new(&scalar, A);
    bml_scale_inplace(&scalar, A);

    trace_mult = bml_trace_mult(A, B);
    trace = scalar * scalar * N;

    if ((diff = fabs(trace - trace_mult) / (double) N) > REL_TOL)
    {
        LOG_ERROR
            ("trace_mult is not correct; trace = %e and not %e, diff = %e\n",
             trace_mult, trace, diff);
        return -1;
    }

    LOG_INFO("test_trace_mult passed\n");

    bml_deallocate(&A);
    bml_deallocate(&B);

    return 0;
}
