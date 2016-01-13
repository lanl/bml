#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "bml_setters_dense.h"
#include "bml_types_dense.h"

void
bml_set_dense(
    bml_matrix_dense_t * A,
    const int i,
    const int j,
    const void *value)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_dense_single_real(A, i, j, value);
            break;
        case double_real:
            bml_set_dense_double_real(A, i, j, value);
            break;
        case single_complex:
            bml_set_dense_single_complex(A, i, j, value);
            break;
        case double_complex:
            bml_set_dense_double_complex(A, i, j, value);
            break;
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}

void
bml_set_row_dense(
    bml_matrix_dense_t * A,
    const int i,
    const void *row)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_row_dense_single_real(A, i, row);
            break;
        case double_real:
            bml_set_row_dense_double_real(A, i, row);
            break;
        case single_complex:
            bml_set_row_dense_single_complex(A, i, row);
            break;
        case double_complex:
            bml_set_row_dense_double_complex(A, i, row);
            break;
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}
