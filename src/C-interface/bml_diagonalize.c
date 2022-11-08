#include "bml_diagonalize.h"
#include "bml_introspection.h"
#include "bml_utilities.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "dense/bml_diagonalize_dense.h"
#include "ellpack/bml_diagonalize_ellpack.h"
#include "ellblock/bml_diagonalize_ellblock.h"
#include "csr/bml_diagonalize_csr.h"
#ifdef BML_USE_MPI
#include "distributed2d/bml_diagonalize_distributed2d.h"
#endif

void
bml_diagonalize(
    bml_matrix_t * A,
    void *eigenvalues,
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
        case ellblock:
            bml_diagonalize_ellblock(A, eigenvalues, eigenvectors);
            break;
        case csr:
            bml_diagonalize_csr(A, eigenvalues, eigenvectors);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            bml_diagonalize_distributed2d(A, eigenvalues, eigenvectors);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }

}
