#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "bml_setters_ellpack.h"
#include "bml_types_ellpack.h"

void
bml_set_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value)
{
    LOG_ERROR("FIXME\n");
}

void
bml_set_row_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    const void *row,
    const double threshold)
{
    switch (A->matrix_precision)
//     switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_row_ellpack_single_real(A, i, row, threshold);
            break;
        case double_real:
            bml_set_row_ellpack_double_real(A, i, row, threshold);
            break;
        case single_complex:
            bml_set_row_ellpack_single_complex(A, i, row, threshold);
            break;
        case double_complex:
            bml_set_row_ellpack_double_complex(A, i, row, threshold);
            break;
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}
