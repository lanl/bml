#include "../bml_trace.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_copy_dense.h"
#include "bml_trace_dense.h"
#include "bml_types_dense.h"

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
double bml_trace_dense(const bml_matrix_dense_t *A)
{
    double trace = 0.0;

    switch(A->matrix_precision) {
    case single_real:

        break;
    case double_real:

        break;
    }
    return trace;
}
