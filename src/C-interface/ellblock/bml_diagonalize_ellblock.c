#include "../bml_logger.h"
#include "../bml_types.h"
#include "../bml_utilities.h"
#include "../dense/bml_types_dense.h"
#include "bml_diagonalize_ellblock.h"
#include "bml_types_ellblock.h"

#include <string.h>

/** \page diagonalize
 *
 */

void
bml_diagonalize_ellblock(
    bml_matrix_ellblock_t * A,
    void *eigenvalues,
    bml_matrix_t * eigenvectors)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_diagonalize_ellblock_single_real(A, eigenvalues,
                                                 eigenvectors);
            break;
        case double_real:
            bml_diagonalize_ellblock_double_real(A, eigenvalues,
                                                 eigenvectors);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_diagonalize_ellblock_single_complex(A, eigenvalues,
                                                    eigenvectors);
            break;
        case double_complex:
            bml_diagonalize_ellblock_double_complex(A, eigenvalues,
                                                    eigenvectors);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
