#include "../bml_logger.h"
#include "bml_getters_distributed2d.h"

void *
bml_get_row_distributed2d(
    bml_matrix_distributed2d_t * A,
    int i)
{
    switch (A->matrix_precision)
    {
        case single_real:
            return bml_get_row_distributed2d_single_real(A, i);
            break;
        case double_real:
            return bml_get_row_distributed2d_double_real(A, i);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_get_row_distributed2d_single_complex(A, i);
            break;
        case double_complex:
            return bml_get_row_distributed2d_double_complex(A, i);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision in bml_get_row_distributed2d\n");
            break;
    }
    return NULL;
}
