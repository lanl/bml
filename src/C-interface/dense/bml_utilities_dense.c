#include "../bml_utilities.h"
#include "bml_types_dense.h"
#include "bml_utilities_dense.h"

void
bml_print_bml_matrix_dense(
    const bml_matrix_dense_t * A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u)
{
    bml_print_dense_matrix(A->N, A->matrix_precision, A->matrix, i_l, i_u,
                           j_l, j_u);
}
