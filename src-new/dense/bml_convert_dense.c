#include "bml_allocate_dense.h"
#include "bml_convert_dense.h"

#include <stdlib.h>

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group
 *
 * \param N The number of rows/columns.
 * \param A_dense The dense matrix
 * \param threshold The matrix element magnited threshold
 * \return The bml matrix
 */
bml_matrix_dense_t *bml_convert_from_dense_dense(const int N,
                                                 const double *A_dense,
                                                 const double threshold)
{
    bml_matrix_dense_t *A;
    A = bml_allocate_dense(double_precision, N);
    return A;
}

/** Convert a bml matrix into a dense matrix.
 *
 * \ingroup convert_group
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
double *bml_convert_to_dense_dense(const bml_matrix_dense_t *A)
{
    return NULL;
}
