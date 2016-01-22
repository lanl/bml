#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_setters.h"
#include "dense/bml_setters_dense.h"
#include "ellpack/bml_setters_ellpack.h"

void
bml_set(
    bml_matrix_t * A,
    const int i,
    const int j,
    const void *value)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_set_dense(A, i, j, value);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

void
bml_set_row(
    bml_matrix_t * A,
    const int i,
    const void *row)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_set_row_dense(A, i, row);
            break;
        case ellpack:
            bml_set_row_ellpack(A, i, row);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

void
bml_set_diag(
    bml_matrix_t * A,
    const void *diag)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_set_diag_dense(A,row);
            break;
        case ellpack:
            bml_set_diag_ellpack(A,row);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

