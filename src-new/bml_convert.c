#include "bml_convert.h"

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group
 *
 * \param matrix_type The matrix type
 * \param A The dense matrix
 * \param threshold The matrix element magnited threshold
 * \return The bml matrix
 */
bml_matrix_t *bml_convert_from_dense(const bml_matrix_type_t matrix_type,
                                     const double *A,
                                     const double threshold)
{
    switch(matrix_type) {
    case dense:
        return bml_convert_from_dense_dense(A, threshold);
    default:
        bml_log(BML_ERROR, "unknown matrix type\n");
    }
}

/** Convert a bml matrix into a dense matrix.
 *
 * \ingroup convert_group
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
double *convert_to_dense(const bml_matrix_t *A)
{
    bml_matrix_type_t *matrix_type = A;

    switch(*matrix_type) {
    case dense:
        return convert_to_dense_dense(A);
    default:
        bml_log(BML_ERROR, "unknown matrix type\n");
    }
}
