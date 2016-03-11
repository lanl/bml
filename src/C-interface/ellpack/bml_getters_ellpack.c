#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "bml_getters_ellpack.h"
#include "bml_types_ellpack.h"


void
bml_get_row_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    void *row)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_get_row_ellpack_single_real(A, i, row);
            break;
        case double_real:
            bml_get_row_ellpack_double_real(A, i, row);
            break;
        case single_complex:
            bml_get_row_ellpack_single_complex(A, i, row);
            break;
        case double_complex:
            bml_get_row_ellpack_double_complex(A, i, row);
            break;
        default:
            LOG_ERROR("unkonwn precision in bml_get_row_ellpack\n");
            break;
    }
}
