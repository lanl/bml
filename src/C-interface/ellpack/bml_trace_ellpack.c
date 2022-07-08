#include "../bml_logger.h"
#include "../bml_trace.h"
#include "../bml_types.h"
#include "bml_trace_ellpack.h"
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
    bml_matrix_ellpack_t * A)
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
#ifdef BML_COMPLEX
        case single_complex:
            trace = bml_trace_ellpack_single_complex(A);
            break;
        case double_complex:
            trace = bml_trace_ellpack_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
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
bml_trace_mult_ellpack(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B)
{
    double trace = 0.0;

    switch (A->matrix_precision)
    {
        case single_real:
            trace = bml_trace_mult_ellpack_single_real(A, B);
            break;
        case double_real:
            trace = bml_trace_mult_ellpack_double_real(A, B);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            trace = bml_trace_mult_ellpack_single_complex(A, B);
            break;
        case double_complex:
            trace = bml_trace_mult_ellpack_double_complex(A, B);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return trace;
}
