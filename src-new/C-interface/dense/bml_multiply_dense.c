#include "bml_multiply.h"
#include "bml_types.h"
#include "bml_multiply_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

/** Matrix multiply.
 *
 * C = alpha * A * B + beta * C
 *
 *  \ingroup multiply_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param C Matrix C
 *  \param alpha Scalar factor multiplied by A * B
 *  \param beta Scalar factor multiplied by C
 */
void bml_multiply_dense(const bml_matrix_dense_t *A, const bml_matrix_dense_t *B, const bml_matrix_dense_t *C, const double alpha, const double beta)
{
    switch(A->matrix_precision) {
    case single_real:
        bml_multiply_dense_single_real(A, B, C, alpha, beta);
        break;
    case double_real:
        bml_multiply_dense_double_real(A, B, C, alpha, beta);
        break;
    }
}
