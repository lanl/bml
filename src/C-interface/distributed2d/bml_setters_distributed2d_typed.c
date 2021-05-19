#include "../../macros.h"
#include "../../typed.h"
#include "../bml_logger.h"
#include "../bml_setters.h"
#include "bml_types_distributed2d.h"

void TYPED_FUNC(
    bml_set_diagonal_distributed2d) (
    bml_matrix_distributed2d_t * A,
    void *_diagonal,
    double threshold)
{
    REAL_T *diagonal = _diagonal;

    if (A->myprow == A->mypcol)
    {
        int offset = A->myprow * A->N / A->nprows;
        bml_set_diagonal(A->matrix, diagonal + offset, threshold);
    }
}
