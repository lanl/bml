#include "bml_getters_ellpack.h"
#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "bml_types_ellpack.h"

void *
bml_get_element_ellpack(
    bml_matrix_ellpack_t * A,
    int i,
    int j)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            return bml_get_element_ellpack_single_real(A, i, j);
            break;
        case double_real:
            return bml_get_element_ellpack_double_real(A, i, j);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_get_element_ellpack_single_complex(A, i, j);
            break;
        case double_complex:
            return bml_get_element_ellpack_double_complex(A, i, j);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision in bml_get_element_ellpack\n");
            break;
    }
    return NULL;
}

void *
bml_get_row_ellpack(
    bml_matrix_ellpack_t * A,
    int i)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            return bml_get_row_ellpack_single_real(A, i);
            break;
        case double_real:
            return bml_get_row_ellpack_double_real(A, i);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_get_row_ellpack_single_complex(A, i);
            break;
        case double_complex:
            return bml_get_row_ellpack_double_complex(A, i);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision in bml_get_row_ellpack\n");
            break;
    }
    return NULL;
}

void *
bml_get_diagonal_ellpack(
    bml_matrix_ellpack_t * A)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            return bml_get_diagonal_ellpack_single_real(A);
            break;
        case double_real:
            return bml_get_diagonal_ellpack_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_get_diagonal_ellpack_single_complex(A);
            break;
        case double_complex:
            return bml_get_diagonal_ellpack_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision in bml_get_diagonal_ellpack\n");
            break;
    }
    return NULL;
}
