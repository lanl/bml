#include "bml_logger.h"
#include "bml_trace.h"
#include "bml_trace_ellpack.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

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
bml_trace_ellpack(
    const bml_matrix_ellpack_t * A)
{
    double trace = 0.0;

    switch (A->matrix_precision)
    {
    case single_real:
        trace = bml_trace_ellpack_single_real(A);
        break;
    case double_real:
        trace = bml_trace_ellpack_double_real(A);
        break;
    default:
        LOG_ERROR("unknown precision\n");
        break;
    }
    return trace;
}
