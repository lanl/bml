#include "bml_transpose.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_transpose_dense.h"
#include "ellpack/bml_transpose_ellpack.h"
#include "ellblock/bml_transpose_ellblock.h"
#include "csr/bml_transpose_csr.h"
#ifdef BML_USE_MPI
#include "distributed2d/bml_transpose_distributed2d.h"
#endif

#include <stdlib.h>

/** Transpose matrix.
 *
 * \ingroup transpose_group_C
 *
 * \param A Matrix to be transposed
 * \return  Transposed A
 */
bml_matrix_t *
bml_transpose_new(
    bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_transpose_new_dense(A);
            break;
        case ellpack:
            return bml_transpose_new_ellpack(A);
            break;
        case ellblock:
            return bml_transpose_new_ellblock(A);
            break;
        case csr:
            return bml_transpose_new_csr(A);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            return bml_transpose_new_distributed2d(A);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return NULL;
}

/** Transpose matrix.
 *
 * \ingroup transpose_group_C
 *
 * \param A Matrix to be transposed
 * \return  Transposed A
 */
void
bml_transpose(
    bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_transpose_dense(A);
            break;
        case ellpack:
            bml_transpose_ellpack(A);
            break;
        case ellblock:
            bml_transpose_ellblock(A);
            break;
        case csr:
            bml_transpose_csr(A);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            bml_transpose_distributed2d(A);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
