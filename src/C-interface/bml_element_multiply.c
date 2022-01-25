#include "bml_element_multiply.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_element_multiply_dense.h"
#include "ellpack/bml_element_multiply_ellpack.h"
#include "ellsort/bml_element_multiply_ellsort.h"
#include "csr/bml_element_multiply_csr.h"
#ifdef DO_MPI
#include "distributed2d/bml_multiply_distributed2d.h"
#endif

#include <stdlib.h>

/** Element-wise Matrix multiply (Hadamard product)
 *
 * \f$ C_{ij} \leftarrow A_{ij} * B_{ij} \f$
 *
 * \ingroup multiply_group_C
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param threshold Threshold for multiplication
 */
void
bml_element_multiply_AB(
    bml_matrix_t * A,
    bml_matrix_t * B,
    bml_matrix_t * C,
    double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_element_multiply_AB_dense(A, B, C);
            break;
        case ellpack:
            bml_element_multiply_AB_ellpack(A, B, C, threshold);
            break;
        case ellsort:
            bml_element_multiply_AB_ellsort(A, B, C, threshold);
            break;
        case csr:
            bml_element_multiply_AB_csr(A, B, C, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
