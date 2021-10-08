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

void TYPED_FUNC(
    bml_set_row_distributed2d) (
    bml_matrix_distributed2d_t * A,
    int i,
    void *row,
    double threshold)
{
    const int nloc = A->N / A->nprows;

    if (i < A->myprow * nloc)
        return;
    if (i >= (A->myprow + 1) * nloc)
        return;

    // subrow corresponds to local columns only
    // by pointing to first local element
    // assuming row is a dense array
    REAL_T *sub_row = (REAL_T *) row + A->mypcol * nloc;

    int irow = i - A->myprow * nloc;
    bml_set_row(A->matrix, irow, sub_row, threshold);
}
