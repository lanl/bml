#include "../../macros.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_types.h"
#include "bml_allocate_csr.h"
#include "bml_import_csr.h"
#include "bml_types_csr.h"
#include "bml_setters_csr.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Convert a dense matrix into a bml csr matrix.
 *
 * \ingroup convert_group
 *
 * \param N The number of rows/columns
 * \param matrix_precision The real precision
 * \param A The dense matrix
 * \return The bml matrix
 */
bml_matrix_csr_t
    * TYPED_FUNC(bml_import_from_dense_csr) (bml_dense_order_t order,
                                             int N, void *A,
                                             double threshold, int M,
                                             bml_distribution_mode_t
                                             distrib_mode)
{
    bml_matrix_csr_t *csr_A =
        TYPED_FUNC(bml_zero_matrix_csr) (N, M, distrib_mode);

    REAL_T *dense_A = (REAL_T *) A;

#pragma omp parallel for shared(dense_A)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            REAL_T A_ij;
            switch (order)
            {
                case dense_row_major:
                    A_ij = dense_A[ROWMAJOR(i, j, N, N)];
                    break;
                case dense_column_major:
                    A_ij = dense_A[COLMAJOR(i, j, N, N)];
                    break;
                default:
                    LOG_ERROR("unknown order\n");
                    break;
            }
            if (is_above_threshold(A_ij, threshold))
            {
                TYPED_FUNC(csr_set_row_element_new) (csr_A->data_[i], j,
                                                     &A_ij);
            }
        }
    }

    return csr_A;
}
