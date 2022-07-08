#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "bml_setters_csr.h"
#include "bml_types_csr.h"


void
bml_set_element_new_csr(
    bml_matrix_csr_t * A,
    const int i,
    const int j,
    void *value)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_element_new_csr_single_real(A, i, j, value);
            break;
        case double_real:
            bml_set_element_new_csr_double_real(A, i, j, value);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_set_element_new_csr_single_complex(A, i, j, value);
            break;
        case double_complex:
            bml_set_element_new_csr_double_complex(A, i, j, value);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision in bml_set_element_new_csr\n");
            break;
    }
}


void
bml_set_element_csr(
    bml_matrix_csr_t * A,
    const int i,
    const int j,
    void *value)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_element_csr_single_real(A, i, j, value);
            break;
        case double_real:
            bml_set_element_csr_double_real(A, i, j, value);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_set_element_csr_single_complex(A, i, j, value);
            break;
        case double_complex:
            bml_set_element_csr_double_complex(A, i, j, value);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision in bml_set_element_csr\n");
            break;
    }
}

void
bml_set_row_csr(
    bml_matrix_csr_t * A,
    const int i,
    void *row,
    const double threshold)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_set_row_csr_single_real(A, i, row, threshold);
            break;
        case double_real:
            bml_set_row_csr_double_real(A, i, row, threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_set_row_csr_single_complex(A, i, row, threshold);
            break;
        case double_complex:
            bml_set_row_csr_double_complex(A, i, row, threshold);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}

void
bml_set_diagonal_csr(
    bml_matrix_csr_t * A,
    void *diagonal,
    const double threshold)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_set_diagonal_csr_single_real(A, diagonal, threshold);
            break;
        case double_real:
            bml_set_diagonal_csr_double_real(A, diagonal, threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_set_diagonal_csr_single_complex(A, diagonal, threshold);
            break;
        case double_complex:
            bml_set_diagonal_csr_double_complex(A, diagonal, threshold);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}

void
bml_set_sparse_row_csr(
    bml_matrix_csr_t * A,
    const int i,
    const int count,
    const int *cols,
    void *row,
    const double threshold)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_set_sparse_row_csr_single_real(A, i, count, cols, row,
                                               threshold);
            break;
        case double_real:
            bml_set_sparse_row_csr_double_real(A, i, count, cols, row,
                                               threshold);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_set_sparse_row_csr_single_complex(A, i, count, cols, row,
                                                  threshold);
            break;
        case double_complex:
            bml_set_sparse_row_csr_double_complex(A, i, count, cols, row,
                                                  threshold);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}
