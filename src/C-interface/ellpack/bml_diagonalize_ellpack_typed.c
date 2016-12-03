#include "../macros.h"
#include "../typed.h"
#include "bml_diagonalize.h"
#include "bml_diagonalize_ellpack.h"
#include "bml_allocate.h"
#include "bml_allocate_ellpack.h"
#include "dense/bml_allocate_dense.h"
#include "bml_convert.h"
#include "bml_convert_ellpack.h"
#include "dense/bml_convert_dense.h"
#include "dense/bml_diagonalize_dense.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"
#include "dense/bml_types_dense.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/** Diagonalize matrix.
 *
 *  \ingroup diag_group
 *
 *  \param A The matrix A
 *  \param eigenvalues Eigenvalues of A
 *  \param eigenvectors Eigenvectors of A
 *  \return The sum of squares of A
 */
void TYPED_FUNC(
    bml_diagonalize_ellpack) (
    const bml_matrix_ellpack_t * A,
    double *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
    double threshold = 0.0;
    bml_matrix_dense_t *D;
    REAL_T *A_dense;

    // Need to convert to dense for ELLPACK
    A_dense = bml_convert_to_dense_ellpack(A, dense_row_major);
    D = bml_convert_from_dense_dense(A->matrix_precision, dense_row_major,
                                     A->N, A_dense, threshold,
                                     A->distribution_mode);

    TYPED_FUNC(bml_diagonalize_dense) (D, eigenvalues, eigenvectors);

    bml_deallocate_dense(D);
    free(A_dense);
}
