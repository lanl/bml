#include "../../macros.h"
#include "../../typed.h"
#include "../bml_add.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_element_multiply.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_add_csr.h"
#include "bml_allocate_csr.h"
#include "bml_element_multiply_csr.h"
#include "bml_types_csr.h"
#include "bml_setters_csr.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Element-wise Matrix multiply (Hadamard product)
 *
 * \f$ C_{ij} \leftarrow A_{ij} * B_{ij} \f$
 *
 * \ingroup multiply_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param threshold Used for sparse multiply
 */
void TYPED_FUNC(
    bml_element_multiply_AB_csr) (
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    bml_matrix_csr_t * C,
    double threshold)
{
    const int A_N = A->N_;
    const int C_N = C->N_;

#if !(defined(__IBMC__) || defined(__ibmxl__))
    int ix[C_N], jx[C_N];
    REAL_T x[C_N];

    memset(ix, 0, C_N * sizeof(int));
    memset(jx, 0, C_N * sizeof(int));
    memset(x, 0.0, C_N * sizeof(REAL_T));
#endif

#if defined(__IBMC__) || defined(__ibmxl__)
#pragma omp parallel for                       \
    shared(A_N, C_N)
#else
#pragma omp parallel for                       \
    firstprivate(ix, jx, x)
#endif

    for (int i = 0; i < A_N; i++)
    {
#if defined(__IBMC__) || defined(__ibmxl__)
        int ix[C_N], jx[C_N];
        REAL_T x[C_N];

        memset(ix, 0, C_N * sizeof(int));
#endif
        int *acols = A->data_[i]->cols_;
        REAL_T *avals = (REAL_T *) A->data_[i]->vals_;
        const int annz = A->data_[i]->NNZ_;
        int l = 0;
        for (int pos = 0; pos < annz; pos++)
        {
            REAL_T a = avals[pos];
            const int k = acols[pos];
            if (ix[k] == 0)
            {
                x[k] = 0.0;
                jx[l] = k;
                ix[k] = i + 1;
                l++;
            }

            const int bnnz = B->data_[i]->NNZ_;
            REAL_T *bvals = (REAL_T *) B->data_[i]->vals_;
            int *bcols = B->data_[i]->cols_;
            for (int bpos = 0; bpos < bnnz; bpos++)
            {
                const int kb = bcols[bpos];
                if (k == kb)
                {
                    x[k] += a * bvals[bpos];
                }
            }
        }

        // clear row
        TYPED_FUNC(csr_clear_row) (C->data_[i]);
        for (int j = 0; j < l; j++)
        {
            int jp = jx[j];
            REAL_T xtmp = x[jp];
            if (jp == i)
            {
                TYPED_FUNC(csr_set_row_element_new) (C->data_[i], jp, &xtmp);
            }
            else if (is_above_threshold(xtmp, threshold))
            {
                TYPED_FUNC(csr_set_row_element_new) (C->data_[i], jp, &xtmp);
            }
            // reset
            ix[jp] = 0;
            x[jp] = 0.0;
        }
    }
}
