#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "bml_getters_dense.h"
#include "bml_types_dense.h"

/*
void
bml_get_dense(
    bml_matrix_dense_t * A,
    const int i,
    const int j,
    const void *value)
{
    LOG_ERROR("FIXME\n");
}
*/

void
bml_get_row_dense(
    bml_matrix_dense_t * A,
    const int i,
    void *row)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_get_row_dense_single_real(A, i, row);
            break;
        case double_real:
            bml_get_row_dense_double_real(A, i, row);
            break;
        case single_complex:
            bml_get_row_dense_single_complex(A, i, row);
            break;
        case double_complex:
            bml_get_row_dense_double_complex(A, i, row);
            break;
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}
