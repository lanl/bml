#include "bml_convert.h"
#include "bml_logger.h"
#include "dense/bml_convert_dense.h"
#include "ellpack/bml_convert_ellpack.h"
#include "ellsort/bml_convert_ellsort.h"
#include "ellblock/bml_convert_ellblock.h"
#include "csr/bml_convert_csr.h"

#include <stdlib.h>

/** Convert a bml matrix to another type.
 *
 * \f$ A \rightarrow B \f$
 *
 * \param A The input matrix.
 * \return The converted matrix \f$ B \f$.
 */
bml_matrix_t *
bml_convert(
    bml_matrix_t * A,
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    switch (matrix_type)
    {
        case dense:
            return bml_convert_dense(A, matrix_precision, distrib_mode);
            break;
        case ellpack:
            return bml_convert_ellpack(A, matrix_precision, M, distrib_mode);
            break;
        case ellsort:
            return bml_convert_ellsort(A, matrix_precision, M, distrib_mode);
            break;
        case ellblock:
            return bml_convert_ellblock(A, matrix_precision, M, distrib_mode);
            break;
        case csr:
            return bml_convert_csr(A, matrix_precision, M, distrib_mode);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return NULL;
}
