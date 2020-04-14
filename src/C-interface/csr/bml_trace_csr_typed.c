#include "../../macros.h"
#include "../../typed.h"
#include "../bml_trace.h"
#include "bml_trace_csr.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_types_csr.h"
#include "../bml_logger.h"
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
    bml_trace_mult_csr) (
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B)
{

    const int A_N = A->N_;
    REAL_T trace = 0.0;
    
    if (A_N != B->N_)
    {
        LOG_ERROR
            ("bml_trace_mult_csr: Matrices A and B have different sizes.");
    }        

#pragma omp parallel for                       \
    shared(A_N)  \
  reduction(+:trace)    

    for (int i = 0; i < A_N; i++)
    {
        int *acols = A->data_[i]->cols_;
        REAL_T *avals = (REAL_T *)A->data_[i]->vals_;
        const int annz = A->data_[i]->NNZ_; 
        
        for (int pos = 0; pos < annz; pos++)
        {
            REAL_T a = avals[pos];
            const int j = acols[pos];

            const int bnnz = B->data_[j]->NNZ_;
            REAL_T* bvals = (REAL_T*)B->data_[j]->vals_;
            int *bcols = B->data_[j]->cols_;            
            for (int bpos = 0; bpos < bnnz; bpos++)
            {
                const int k = bcols[bpos];
                if(i == k)
                {
                    trace = trace + a * bvals[bpos];
                    break;
                }
            }
        }
    } 
    return trace;
}
