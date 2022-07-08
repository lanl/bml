#include "../bml_add.h"
#include "../bml_logger.h"
#include "../bml_element_multiply.h"
#include "../bml_types.h"
#include "bml_add_ellsort.h"
#include "bml_element_multiply_ellsort.h"
#include "bml_types_ellsort.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/** Element-wise Matrix multiply (Hadamard product)
 *
 * \f$ C_{ij} \leftarrow A_{ij} * B_{ij} \f$
 *
 *  \ingroup multiply_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param C Matrix C
 *  \param threshold Used for sparse multiply
 */
void
bml_element_multiply_AB_ellsort(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    double threshold)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_element_multiply_AB_ellsort_single_real(A, B, C, threshold);
            break;
        case double_real:
            bml_element_multiply_AB_ellsort_double_real(A, B, C, threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_element_multiply_AB_ellsort_single_complex(A, B, C,
                                                           threshold);
            break;
        case double_complex:
            bml_element_multiply_AB_ellsort_double_complex(A, B, C,
                                                           threshold);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
