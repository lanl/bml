#include "bml_convert.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_convert_dense.h"
#include "ellpack/bml_convert_ellpack.h"

#include <stdlib.h>

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group_C
 *
 * \param matrix_type The matrix type
 * \param matrix_precision The real precision
 * \param N The number of rows/columns
 * \param A The dense matrix
 * \param threshold The matrix element magnited threshold
 * \param M The number of non-zeroes per row
 * \return The bml matrix
 */
bml_matrix_t *
bml_convert_from_dense (const bml_matrix_type_t matrix_type,
                        const bml_matrix_precision_t matrix_precision,
                        const int N,
                        const void *A, const double threshold, const int M)
{
    LOG_DEBUG ("Converting dense matrix to bml format\n");
    switch (matrix_type)
    {
    case dense:
        return bml_convert_from_dense_dense (matrix_precision, N, A,
                                             threshold);
    case ellpack:
        return bml_convert_from_dense_ellpack (matrix_precision, N, A,
                                               threshold, M);
    default:
        LOG_ERROR ("unknown matrix type\n");
    }
    return NULL;
}

/** Convert a bml matrix into a dense matrix.
 *
 * The returned pointer has to be typecase into the proper real
 * type. If the bml matrix is a single precision matrix, then the
 * following should be used:
 *
 * \code{.c}
 * float *A_dense = bml_convert_to_dense(A_bml);
 * \endcode
 *
 * The matrix size can be queried with
 *
 * \code{.c}
 * int N = bml_get_size(A_bml);
 * \endcode
 *
 * \ingroup convert_group_C
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
void *
bml_convert_to_dense (const bml_matrix_t * A)
{
    LOG_DEBUG ("Converting bml matrix to dense\n");
    switch (bml_get_type (A))
    {
    case dense:
        return bml_convert_to_dense_dense (A);
    case ellpack:
        return bml_convert_to_dense_ellpack (A);
    default:
        LOG_ERROR ("unknown matrix type\n");
    }
    return NULL;
}
