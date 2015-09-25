#include "../typed.h"
#include "bml_trace.h"
#include "bml_types.h"
#include "bml_trace_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>

/** Calculate the trace of a matrix.
 *
 *  \ingroup trace_group
 *
 *  \param A The matrix to calculate a trace for
 *  \return the trace of A
 */
double TYPED_FUNC(bml_trace_ellpack)(const bml_matrix_ellpack_t *A)
{
    double trace = 0.0;

    REAL_T *A_value = (REAL_T*)A->value;

    int N = A->N;
    int M = A->M;

    #pragma omp parallel for reduction(+:trace)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < A->nnz[i]; j++)
        {
          if (A->index[j+i*M] == i) trace += A_value[j+i*M];
        }
    }

    return trace;
}
