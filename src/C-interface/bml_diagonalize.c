#include "bml_diagonalize.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "dense/bml_diagonalize_dense.h"
#include "ellpack/bml_diagonalize_ellpack.h"

void
bml_diagonalize(
    const bml_matrix_t * A,
    double *eigenvalues,
    bml_matrix_t * eigenvectors)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_diagonalize_dense(A, eigenvalues, eigenvectors);
            break;
        case ellpack:
            bml_diagonalize_ellpack(A, eigenvalues, eigenvectors);
            break;
        case ellsort:
            LOG_ERROR("diagonalize routine is not implemented for ellsort\n");
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
