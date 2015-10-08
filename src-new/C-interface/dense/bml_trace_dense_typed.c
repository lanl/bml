#include "../typed.h"
#include "../blas.h"
#include "bml_trace.h"
#include "bml_types.h"
#include "bml_trace_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    double trace = 0.0;
    int N = A->N;
    REAL_T *A_matrix = A->matrix;

    #pragma omp parallel for reduction(+:trace)
    for (int i = 0; i < N; i++)
    {
        trace += A_matrix[i + i * N];
    }

    return trace;
}
