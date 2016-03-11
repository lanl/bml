#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_getters.h"
#include "dense/bml_getters_dense.h"
#include "ellpack/bml_getters_ellpack.h"
/*
void
bml_get(
    bml_matrix_t * A,
    const int i,
    const int j,
    void *value)
{
    LOG_ERROR("FIXME\n");
}
*/

void
bml_get_row(
    bml_matrix_t * A,
    const int i,
    void *row)
{
    switch (bml_get_type(A))
    {
        case dense:          
            bml_get_row_dense(A, i, row);
            break;
        case ellpack:
            bml_get_row_ellpack(A, i, row);
            break;            
        default:            
            LOG_ERROR("unknown matrix type in bml_get_row\n");
            break;
    }
}
