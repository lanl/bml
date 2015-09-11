#include "bml_convert.h"
#include "bml_logger.h"
#include "dense/bml_convert_dense.h"

#include <stdlib.h>

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group
 *
 * \param matrix_type The matrix type
 * \param matrix_precision The real precision
 * \param N The number of rows/columns
 * \param A The dense matrix
 * \param threshold The matrix element magnited threshold
 * \return The bml matrix
 */
bml_matrix_t *bml_convert_from_dense(const bml_matrix_type_t matrix_type,
                                     const bml_matrix_precision_t matrix_precision,
                                     const int N,
                                     const void *A,
                                     const double threshold)
{
    switch(matrix_type) {
    case dense:
        return bml_convert_from_dense_dense(matrix_precision, N, A, threshold);
    default:
        LOG_ERROR("unknown matrix type\n");
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
void *bml_convert_to_dense(const bml_matrix_t *A)
{
    const bml_matrix_type_t *matrix_type = A;

    switch(*matrix_type) {
    case dense:
        return bml_convert_to_dense_dense(A);
    default:
        LOG_ERROR("unknown matrix type\n");
    }
    return NULL;
}
