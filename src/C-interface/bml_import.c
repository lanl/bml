#include "bml_import.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_convert_dense.h"
#include "ellpack/bml_convert_ellpack.h"
#include "ellsort/bml_convert_ellsort.h"

#include <stdlib.h>

/** Import a dense matrix.
 *
 * \ingroup convert_group_C
 *
 * \param matrix_type The matrix type
 * \param matrix_precision The real precision
 * \param order The dense matrix element order
 * \param N The number of rows/columns
 * \param A The dense matrix
 * \param threshold The matrix element magnited threshold
 * \param M The number of non-zeroes per row
 * \return The bml matrix
 */
bml_matrix_t *
bml_import_from_dense(
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const bml_dense_order_t order,
    const int N,
    const void *A,
    const double threshold,
    const int M,
    const bml_distribution_mode_t distrib_mode)
{
    LOG_DEBUG("importing dense matrix\n");
    switch (matrix_type)
    {
        case dense:
            return bml_convert_from_dense_dense(matrix_precision, order, N, A,
                                                threshold, distrib_mode);
        case ellpack:
            return bml_convert_from_dense_ellpack(matrix_precision, order, N,
                                                  A, threshold, M, distrib_mode);
        case ellsort:
            return bml_convert_from_dense_ellsort(matrix_precision, order, N,
                                                  A, threshold, M, distrib_mode);
        default:
            LOG_ERROR("unknown matrix type\n");
    }
    return NULL;
}

/** \deprecated Deprecated API.
 */
bml_matrix_t *
bml_convert_from_dense(
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const bml_dense_order_t order,
    const int N,
    const void *A,
    const double threshold,
    const int M,
    const bml_distribution_mode_t distrib_mode)
{
    return bml_import_from_dense(matrix_type, matrix_precision, order, N, A,
                                 threshold, M, distrib_mode);
}
