#include "bml_export.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_convert_dense.h"
#include "ellpack/bml_convert_ellpack.h"

#include <stdlib.h>

/** Export a bml matrix.
 *
 * The returned pointer has to be typecase into the proper real
 * type. If the bml matrix is a single precision matrix, then the
 * following should be used:
 *
 * \code{.c}
 * float *A_dense = bml_convert_to_dense(A_bml);
 * \endcode
 *
 * The matrix size can be queried with
 *
 * \code{.c}
 * int N = bml_get_size(A_bml);
 * \endcode
 *
 * \ingroup convert_group_C
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
void *
bml_export_to_dense(
    const bml_matrix_t * A)
{
    LOG_DEBUG("Exporting bml matrix to dense\n");
    switch (bml_get_type(A))
    {
        case dense:
            return bml_convert_to_dense_dense(A);
        case ellpack:
            return bml_convert_to_dense_ellpack(A);
        default:
            LOG_ERROR("unknown matrix type\n");
    }
    return NULL;
}

/** \deprecated Deprecated API.
 */
void *
bml_convert_to_dense(
    const bml_matrix_t * A)
{
    return bml_export_to_dense(A);
}
