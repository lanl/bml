#include "../../macros.h"
#include "../../typed.h"
#include "bml_trace.h"
#include "bml_trace_csr.h"
#include "bml_parallel.h"
#include "bml_types.h"
#include "bml_types_csr.h"
#include "bml_logger.h"
#include "bml_getters_csr.h"

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
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
    bml_trace_csr) (
    const bml_matrix_csr_t * A)
{
    const int N = A->N_;

/* We currently assume sequential mode */
/*
    int *A_index = (int *) A->index;
    int *A_nnz = (int *) A->nnz;
    int *A_localRowMin = (int *) A->domain->localRowMin;
    int *A_localRowMax = (int *) A->domain->localRowMax;
*/

    REAL_T trace = 0.0;

#pragma omp parallel for default(none)          \
  reduction(+:trace)
    for (int i = 0; i < N; i++)
    {
        trace += *((REAL_T*)TYPED_FUNC(csr_get_row_element)(A->data_[i], i));
    }

    return (double) REAL_PART(trace);
}

/** Calculate the trace of a matrix multiplication.
 * Both matrices must have the same size.
 *
 *  \ingroup trace_group
 *
 *  \param A The matrix A
 *  \param A The matrix B
 *  \return the trace of A*B
 */
double TYPED_FUNC(
    bml_traceMult_csr) (
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B)
{
    LOG_ERROR("bml_traceMult_csr:  Not implemented");
    return 0.;
}
