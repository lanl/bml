#include "bml.h"

/** Allocate a matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_type The matrix type.
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param n The matrix size.
 *  \param a The matrix.
 */
void bml_allocate(const bml_matrix_type_t matrix_type,
                  const bml_matrix_precision_t matrix_precision,
                  bml_matrix_t A,
                  const int N)
{
    switch(matrix_type) {
    case dense:
        bml_allocate_dense(matrix_precision, A, N);
    default:
        bml_log(BML_ERROR, "Unknown matrix type\n");
    }
}
