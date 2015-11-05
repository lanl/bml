#include "../macros.h"
#include "../typed.h"
#include "bml_trace.h"
#include "bml_trace_ellpack.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

#include <complex.h>
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
 *  \return the trace of A
 */
double TYPED_FUNC(
    bml_trace_ellpack) (
    const bml_matrix_ellpack_t * A)
{
    int N = A->N;
    int M = A->M;

    REAL_T trace = 0.0;
    REAL_T *A_value = (REAL_T *) A->value;

#pragma omp parallel for default(none) shared(N,M,A_value) reduction(+:trace)
    for (int i = 0; i < N; i++)
    {
        trace += A_value[ROWMAJOR(i, 0, M)];
    }

    return (double) REAL_PART(trace);
}
