#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "../bml_setters.h"
#include "bml_setters_distributed2d.h"
#include "bml_types_distributed2d.h"

/** Setters for diagonal.
 *
 * \param A The matrix.
 * \param diag The diagonal (a vector with all diagonal elements).
 */
void
bml_set_diagonal_distributed2d(
    bml_matrix_distributed2d_t * A,
    void *diagonal,
    double threshold)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_set_diagonal_distributed2d_single_real(A, diagonal,
                                                       threshold);
            break;
        case double_real:
            bml_set_diagonal_distributed2d_double_real(A, diagonal,
                                                       threshold);
            break;
        case single_complex:
            bml_set_diagonal_distributed2d_single_complex(A, diagonal,
                                                          threshold);
            break;
        case double_complex:
            bml_set_diagonal_distributed2d_double_complex(A, diagonal,
                                                          threshold);
            break;
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}
