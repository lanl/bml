#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "bml_allocate_distributed2d.h"
#include "bml_import_distributed2d.h"
#include "bml_types_distributed2d.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group
 *
 * \param N The number of rows/columns
 * \param matrix_precision The real precision
 * \param A The dense matrix
 * \param threshold The matrix element magnited threshold
 * \return The bml matrix
 */
bml_matrix_distributed2d_t *
bml_import_from_dense_distributed2d(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    bml_dense_order_t order,
    int N,
    void *A,
    double threshold,
    int M)
{
    switch (matrix_precision)
    {
        case single_real:
            return
                bml_import_from_dense_distributed2d_single_real(matrix_type,
                                                                order, N, A,
                                                                threshold, M);
            break;
        case double_real:
            return
                bml_import_from_dense_distributed2d_double_real(matrix_type,
                                                                order, N, A,
                                                                threshold, M);
            break;
        case single_complex:
            return
                bml_import_from_dense_distributed2d_single_complex
                (matrix_type, order, N, A, threshold, M);
            break;
        case double_complex:
            return
                bml_import_from_dense_distributed2d_double_complex
                (matrix_type, order, N, A, threshold, M);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
