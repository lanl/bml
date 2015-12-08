#include "bml_elemental.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "dense/bml_elemental_dense.h"
#include "ellpack/bml_elemental_ellpack.h"

#include <complex.h>

/** Return a single matrix element.
 *
 * \param i The row index
 * \param j The column index
 * \param A The bml matrix
 * \return The matrix element
 */
float
bml_get_single_real(
    const bml_matrix_t * A,
    const int i,
    const int j)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_get_dense_single_real(A, i, j);
            break;
        case ellpack:
            return bml_get_ellpack_single_real(A, i, j);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return -1;
}

/** Return a single matrix element.
 *
 * \param i The row index
 * \param j The column index
 * \param A The bml matrix
 * \return The matrix element
 */
double
bml_get_double_real(
    const bml_matrix_t * A,
    const int i,
    const int j)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_get_dense_double_real(A, i, j);
            break;
        case ellpack:
            return bml_get_ellpack_double_real(A, i, j);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return -1;
}

/** Return a single matrix element.
 *
 * \param i The row index
 * \param j The column index
 * \param A The bml matrix
 * \return The matrix element
 */
float complex
bml_get_single_complex(
    const bml_matrix_t * A,
    const int i,
    const int j)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_get_dense_single_complex(A, i, j);
            break;
        case ellpack:
            return bml_get_ellpack_single_complex(A, i, j);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return -1;
}

/** Return a single matrix element.
 *
 * \param i The row index
 * \param j The column index
 * \param A The bml matrix
 * \return The matrix element
 */
double complex
bml_get_double_complex(
    const bml_matrix_t * A,
    const int i,
    const int j)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_get_dense_double_complex(A, i, j);
            break;
        case ellpack:
            return bml_get_ellpack_double_complex(A, i, j);
            break;

        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return -1;
}
