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
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_ellpack_single_real(A, i, j, value);
            break;
        case double_real:
            bml_set_ellpack_double_real(A, i, j, value);
            break;
        case single_complex:
            bml_set_ellpack_single_complex(A, i, j, value);
            break;
        case double_complex:
            bml_set_ellpack_double_complex(A, i, j, value);
            break;
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}

void
bml_set_row_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    const void *row)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_row_ellpack_single_real(A, i, row);
            break;
        case double_real:
            bml_set_row_ellpack_double_real(A, i, row);
            break;
        case single_complex:
            bml_set_row_ellpack_single_complex(A, i, row);
            break;
        case double_complex:
            bml_set_row_ellpack_double_complex(A, i, row);
            break;
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}

/* Setter for diagonal */
void
bml_set_diag_ellpack(
    bml_matrix_ellpack_t * A,
    const void *)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_diag_ellpack_single_real(A, diag);
            break;
        case double_real:
            bml_set_diag_ellpack_double_real(A, diag);
            break;
        case single_complex:
            bml_set_diag_ellpack_single_complex(A, diag);
            break;
        case double_complex:
            bml_set_diag_ellpack_double_complex(A, diag);
            break;
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}

