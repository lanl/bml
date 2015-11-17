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

    int *A_index = (int *) A->index;
    int *A_nnz = (int *) A->nnz;

    REAL_T trace = 0.0;
    REAL_T *A_value = (REAL_T *) A->value;

#pragma omp parallel for default(none) shared(N, M, A_value, A_index, A_nnz) reduction(+:trace)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < A_nnz[i]; j++)
        {
            if (i == A_index[ROWMAJOR(i, j, N, M)])
            {
                trace += A_value[ROWMAJOR(i, j, N, M)];
                break;
            }
        }
    }

    return (double) REAL_PART(trace);
}
