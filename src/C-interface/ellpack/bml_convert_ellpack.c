#include "bml_allocate.h"
#include "bml_allocate_ellpack.h"
#include "bml_convert_ellpack.h"
#include "bml_logger.h"
#include "bml_types_ellpack.h"

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
bml_matrix_ellpack_t *
bml_import_from_dense_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const bml_dense_order_t order,
    const int N,
    const void *A,
    const double threshold,
    const int M,
    const bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_import_from_dense_ellpack_single_real(order, N, A,
                                                             threshold, M,
                                                             distrib_mode);
            break;
        case double_real:
            return bml_import_from_dense_ellpack_double_real(order, N, A,
                                                             threshold, M,
                                                             distrib_mode);
            break;
        case single_complex:
            return bml_import_from_dense_ellpack_single_complex(order, N, A,
                                                                threshold,
                                                                M,
                                                                distrib_mode);
            break;
        case double_complex:
            return bml_import_from_dense_ellpack_double_complex(order, N, A,
                                                                threshold,
                                                                M,
                                                                distrib_mode);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}

/** Convert a bml matrix into a dense matrix.
 *
 * \ingroup convert_group
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
void *
bml_export_to_dense_ellpack(
    const bml_matrix_ellpack_t * A,
    const bml_dense_order_t order)
{
    switch (A->matrix_precision)
    {
        case single_real:
            return bml_export_to_dense_ellpack_single_real(A, order);
            break;
        case double_real:
            return bml_export_to_dense_ellpack_double_real(A, order);
            break;
        case single_complex:
            return bml_export_to_dense_ellpack_single_complex(A, order);
            break;
        case double_complex:
            return bml_export_to_dense_ellpack_double_complex(A, order);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
