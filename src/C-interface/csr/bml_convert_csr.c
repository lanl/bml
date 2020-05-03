#include "../bml_logger.h"
#include "bml_convert_csr.h"

#include <stdlib.h>

bml_matrix_csr_t *
bml_convert_csr(
    bml_matrix_t * A,
    bml_matrix_precision_t matrix_precision,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_convert_csr_single_real(A, matrix_precision, M,
                                               distrib_mode);
            break;
        case double_real:
            return bml_convert_csr_double_real(A, matrix_precision, M,
                                               distrib_mode);
            break;
        case single_complex:
            return bml_convert_csr_single_complex(A, matrix_precision, M,
                                                  distrib_mode);
            break;
        case double_complex:
            return bml_convert_csr_double_complex(A, matrix_precision, M,
                                                  distrib_mode);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
