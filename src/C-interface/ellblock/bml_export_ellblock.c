#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "bml_allocate_ellblock.h"
#include "bml_export_ellblock.h"
#include "bml_types_ellblock.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/** Convert a bml matrix into a dense matrix.
 *
 * \ingroup convert_group
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
void *
bml_export_to_dense_ellblock(
    bml_matrix_ellblock_t * A,
    bml_dense_order_t order)
{
    switch (A->matrix_precision)
    {
        case single_real:
            return bml_export_to_dense_ellblock_single_real(A, order);
            break;
        case double_real:
            return bml_export_to_dense_ellblock_double_real(A, order);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_export_to_dense_ellblock_single_complex(A, order);
            break;
        case double_complex:
            return bml_export_to_dense_ellblock_double_complex(A, order);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}
