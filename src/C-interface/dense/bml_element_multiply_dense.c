#include "../bml_logger.h"
#include "../bml_element_multiply.h"
#include "../bml_types.h"
#include "bml_element_multiply_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

/** Element-wise Matrix multiply (Hadamard product)
 *
 * \f$ C_{ij} \leftarrow A_{ij} * B_{ij} \f$
 *
 * \ingroup multiply_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 */
void
bml_element_multiply_AB_dense(
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_element_multiply_AB_dense_single_real(A, B, C);
            break;
        case double_real:
            bml_element_multiply_AB_dense_double_real(A, B, C);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_element_multiply_AB_dense_single_complex(A, B, C);
            break;
        case double_complex:
            bml_element_multiply_AB_dense_double_complex(A, B, C);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
