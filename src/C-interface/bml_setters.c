#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_setters.h"
#include "dense/bml_setters_dense.h"

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
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
