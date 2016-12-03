#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "bml_getters_ellsort.h"
#include "bml_types_ellsort.h"


// Getters diagonal

void
bml_get_diagonal_ellsort(
    bml_matrix_ellsort_t * A,
    void *diagonal)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_get_diagonal_ellsort_single_real(A, diagonal);
            break;
        case double_real:
            bml_get_diagonal_ellsort_double_real(A, diagonal);
            break;
        case single_complex:
            bml_get_diagonal_ellsort_single_complex(A, diagonal);
            break;
        case double_complex:
            bml_get_diagonal_ellsort_double_complex(A, diagonal);
            break;
        default:
            LOG_ERROR("unkonwn precision in bml_get_diagonal_ellsort\n");
            break;
    }
}


// Getters for row

void
bml_get_row_ellsort(
    bml_matrix_ellsort_t * A,
    const int i,
    void *row)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_get_row_ellsort_single_real(A, i, row);
            break;
        case double_real:
            bml_get_row_ellsort_double_real(A, i, row);
            break;
        case single_complex:
            bml_get_row_ellsort_single_complex(A, i, row);
            break;
        case double_complex:
            bml_get_row_ellsort_double_complex(A, i, row);
            break;
        default:
            LOG_ERROR("unkonwn precision in bml_get_row_ellsort\n");
            break;
    }
}
