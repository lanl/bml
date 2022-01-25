#include "../bml_add.h"
#include "../bml_logger.h"
#include "../bml_element_multiply.h"
#include "../bml_types.h"
#include "bml_add_csr.h"
#include "bml_element_multiply_csr.h"
#include "bml_types_csr.h"

#include <stdio.h>
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
 * \param threshold Used for sparse multiply
 */
void
bml_element_multiply_AB_csr(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    bml_matrix_csr_t * C,
    double threshold)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_element_multiply_AB_csr_single_real(A, B, C, threshold);
            break;
        case double_real:
            bml_element_multiply_AB_csr_double_real(A, B, C, threshold);
            break;
        case single_complex:
            bml_element_multiply_AB_csr_single_complex(A, B, C, threshold);
            break;
        case double_complex:
            bml_element_multiply_AB_csr_double_complex(A, B, C, threshold);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
