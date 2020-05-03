#include "../../macros.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_types.h"
#include "bml_allocate_csr.h"
#include "bml_export_csr.h"
#include "bml_types_csr.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Convert a bml matrix into a dense matrix.
 *
 * \ingroup convert_group
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
void *TYPED_FUNC(
    bml_export_to_dense_csr) (
    bml_matrix_csr_t * A,
    bml_dense_order_t order)
{
    int N = A->N_;
    REAL_T *A_dense = bml_allocate_memory(sizeof(REAL_T) * N * N);

    switch (order)
    {
        case dense_row_major:
#pragma omp parallel for shared(N, A_dense)
            for (int i = 0; i < N; i++)
            {
                int *cols = A->data_[i]->cols_;
                REAL_T *vals = (REAL_T *) A->data_[i]->vals_;
                const int annz = A->data_[i]->NNZ_;
                for (int pos = 0; pos < annz; pos++)
                {
                    const int j = cols[pos];
                    A_dense[ROWMAJOR(i, j, N, N)] = vals[pos];
                }
            }
            break;
        case dense_column_major:
#pragma omp parallel for shared(N, A_dense)
            for (int i = 0; i < N; i++)
            {
                int *cols = A->data_[i]->cols_;
                REAL_T *vals = (REAL_T *) A->data_[i]->vals_;
                const int annz = A->data_[i]->NNZ_;
                for (int pos = 0; pos < annz; pos++)
                {
                    const int j = cols[pos];
                    A_dense[COLMAJOR(i, j, N, N)] = vals[pos];
                }
            }
            break;
        default:
            LOG_ERROR("unknown order\n");
            break;
    }
    return A_dense;
}
