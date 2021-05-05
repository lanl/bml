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
    if (A->myprow == A->mypcol)
        bml_set_diagonal(A->matrix, diagonal, threshold);
}
