#include "../bml_introspection.h"
#include "../bml_convert.h"
#include "../bml_logger.h"
#include "bml_allocate_distributed2d.h"
#include "bml_convert_distributed2d.h"
#include "bml_introspection_distributed2d.h"

#include <assert.h>

bml_matrix_distributed2d_t *
bml_convert_distributed2d(
    bml_matrix_t * A,
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int M)
{
    assert(A != NULL);
    int N = bml_get_N(A);
    bml_matrix_distributed2d_t *B = bml_zero_matrix_distributed2d(matrix_type,
                                                                  matrix_precision,
                                                                  N,
                                                                  M);
    bml_matrix_t *matrix = bml_get_local_matrix(A);
    int m = bml_get_M(matrix);
    B->matrix = bml_convert(matrix, matrix_type,
                            matrix_precision, m, sequential);

    return B;
}
