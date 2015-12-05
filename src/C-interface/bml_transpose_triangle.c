#include "bml_transpose_triangle.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_transpose_triangle_dense.h"

#include <stdlib.h>

/** Transposes a triangle of a matrix in place.
 *
 *  \ingroup transpose_triangle_group
 *
 *  \param A  The matrix for which the triangle should be transposed
 *  \param triangle  Which triangle to transpose ('u': upper, 'l': lower)
 */
void
bml_transpose_triangle(
    bml_matrix_t * A,
    char triangle)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_transpose_triangle_dense(A, triangle);
            break;
            /*
               case ellpack:
               bml_transpose_triangle_ellpack(A, triangle);
               break;
             */
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
