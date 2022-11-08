#include "bml_trace.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_trace_dense.h"
#include "ellpack/bml_trace_ellpack.h"
#include "ellsort/bml_trace_ellsort.h"
#include "ellblock/bml_trace_ellblock.h"
#include "csr/bml_trace_csr.h"
#ifdef BML_USE_MPI
#include "distributed2d/bml_trace_distributed2d.h"
#endif

#include <stdlib.h>
/** Calculate trace of a matrix.
 *
 * \ingroup trace_group_C
 *
 * \param A Matrix tocalculate trace for
 * \return  Trace of A
 */
double
bml_trace(
    bml_matrix_t * A)
{
    LOG_DEBUG("bml_trace\n");
    switch (bml_get_type(A))
    {
        case dense:
            return bml_trace_dense(A);
            break;
        case ellpack:
            return bml_trace_ellpack(A);
            break;
        case ellsort:
            return bml_trace_ellsort(A);
            break;
        case ellblock:
            return bml_trace_ellblock(A);
            break;
        case csr:
            return bml_trace_csr(A);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            return bml_trace_distributed2d(A);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return 0;
}

/** Calculate trace of a matrix multiplication.
 *
 * \ingroup trace_group_C
 *
 * \param A Matrix A
 * \param B Matrix B
 * \return  Trace of A*B
 */
double
bml_trace_mult(
    bml_matrix_t * A,
    bml_matrix_t * B)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_trace_mult_dense(A, B);
            break;
        case ellpack:
            return bml_trace_mult_ellpack(A, B);
            break;
        case ellsort:
            return bml_trace_mult_ellsort(A, B);
            break;
        case ellblock:
            return bml_trace_mult_ellblock(A, B);
            break;
        case csr:
            return bml_trace_mult_csr(A, B);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            return bml_trace_mult_distributed2d(A, B);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return 0;
}
