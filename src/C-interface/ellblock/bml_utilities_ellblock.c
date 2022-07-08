#include "../bml_utilities.h"
#include "../bml_logger.h"
#include "bml_types_ellblock.h"
#include "bml_utilities_ellblock.h"

void
bml_read_bml_matrix_ellblock(
    bml_matrix_ellblock_t * A,
    char *filename)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_read_bml_matrix_ellblock_single_real(A, filename);
            break;
        case double_real:
            bml_read_bml_matrix_ellblock_double_real(A, filename);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_read_bml_matrix_ellblock_single_complex(A, filename);
            break;
        case double_complex:
            bml_read_bml_matrix_ellblock_double_complex(A, filename);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_write_bml_matrix_ellblock(
    bml_matrix_ellblock_t * A,
    char *filename)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_write_bml_matrix_ellblock_single_real(A, filename);
            break;
        case double_real:
            bml_write_bml_matrix_ellblock_double_real(A, filename);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_write_bml_matrix_ellblock_single_complex(A, filename);
            break;
        case double_complex:
            bml_write_bml_matrix_ellblock_double_complex(A, filename);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
