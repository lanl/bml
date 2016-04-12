#include "../macros.h"
#include "../blas.h"
#include "../typed.h"
#include "bml_trace.h"
#include "bml_trace_dense.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/** Calculate the trace of a matrix.
 *
 *  \ingroup trace_group
 *
 *  \param A The matrix to calculate a trace for
 *  \return The trace of A
 */
double TYPED_FUNC(
    bml_trace_dense) (
    const bml_matrix_dense_t * A)
{
    int N = A->N;

    REAL_T trace = 0.0;
    REAL_T *A_matrix = A->matrix;

#pragma omp parallel for default(none) shared(N, A_matrix) reduction(+:trace)
    for (int i = 0; i < N; i++)
    {
        trace += A_matrix[ROWMAJOR(i, i, N, N)];
    }

    return (double) REAL_PART(trace);
}
