#include "../bml_utilities.h"
#include "../bml_logger.h"
#include "bml_types_dense.h"
#include "bml_utilities_dense.h"

void
bml_print_bml_matrix_dense(
    const bml_matrix_dense_t * A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u)
{
    bml_print_dense_matrix(A->N, A->matrix_precision, dense_row_major,
                           A->matrix, i_l, i_u, j_l, j_u);
}

void
bml_read_bml_matrix_dense(
    const bml_matrix_dense_t * A,
    const char * filename)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_read_bml_matrix_dense_single_real(A, filename);
            break;
        case double_real:
            bml_read_bml_matrix_dense_double_real(A, filename);
            break;
        case single_complex:
            bml_read_bml_matrix_dense_single_complex(A, filename);
            break;
        case double_complex:
            bml_read_bml_matrix_dense_double_complex(A, filename);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_write_bml_matrix_dense(
    const bml_matrix_dense_t * A,
    const char * filename)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_write_bml_matrix_dense_single_real(A, filename);
            break;
        case double_real:
            bml_write_bml_matrix_dense_double_real(A, filename);
            break;
        case single_complex:
            bml_write_bml_matrix_dense_single_complex(A, filename);
            break;
        case double_complex:
            bml_write_bml_matrix_dense_double_complex(A, filename);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

