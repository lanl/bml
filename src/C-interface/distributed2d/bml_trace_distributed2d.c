#include "../bml_logger.h"
#include "../bml_trace.h"
#include "../bml_types.h"
#include "../bml_parallel.h"
#include "bml_trace_distributed2d.h"
#include "bml_types_distributed2d.h"

#include <stdlib.h>
#include <string.h>

/** Calculate the trace of a matrix.
 *
 *  \ingroup trace_group
 *
 *  \param A The matrix to calculate a trace for
 *  \return the trace of A
 */
double
bml_trace_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    double trace = 0.;

    if (A->myprow == A->mypcol)
        trace = bml_trace(A->matrix);

    bml_sumRealReduce(&trace);

    return trace;
}

/** Calculate the trace of a matrix multiplication.
 * Both matrices must have the same size.
 *
 *  \ingroup trace_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \return the trace of A*B
 */
double
bml_trace_mult_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B)
{
    double trace = bml_trace_mult(A->matrix, B->matrix);;

    bml_sumRealReduce(&trace);

    return trace;
}
