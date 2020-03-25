#include "../../macros.h"
#include "../../typed.h"
#include "../bml_introspection.h"
#include "../bml_types.h"
#include "bml_introspection_csr.h"
#include "bml_types_csr.h"

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

/** Return the sparsity of a matrix.
 *
 *  Note that the the sparsity of a matrix is defined
 *  as (1 - NumberOfZeroes/N*N) where N is the matrix dimension.
 *
 * \ingroup introspection_group_C
 *
 * \param A The bml matrix.
 * \param threshold The threshold used to compute the sparsity.
 * \return The sparsity of A.
 */
double TYPED_FUNC(
    bml_get_sparsity_csr) (
    bml_matrix_csr_t * A,
    double threshold)
{
    int nnzs = 0;
    int i;
    double sparsity;
    int N = A->N_;

    for (int i = 0; i < N; i++)
    {
         int *cols = A->data_[i]->cols_;
        REAL_T *vals = (REAL_T *)A->data_[i]->vals_;
        const int annz = A->data_[i]->NNZ_;  
        for (int pos = 0; pos < annz; pos++) 
        {
            if (ABS(vals[pos])> threshold)
            {
                nnzs++;
            }
        }
    }

    sparsity = (1.0 - (double) nnzs / ((double) (N * N)));
    return sparsity;
}
