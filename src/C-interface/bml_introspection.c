#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "dense/bml_introspection_dense.h"
#include "ellpack/bml_introspection_ellpack.h"

#include <stdlib.h>

/** Returns the matrix type.
 *
 * If the matrix is not initialized yet, a type of "unitialized" is returned.
 *
 * \param A The matrix.
 * \return The matrix type.
 */
bml_matrix_type_t
bml_get_type(
    const bml_matrix_t * A)
{
    const bml_matrix_type_t *matrix_type = A;
    if (A != NULL)
    {
        return *matrix_type;
    }
    else
    {
        return type_uninitialized;
    }
}

/** Return the matrix precision.
 *
 * \param A The matrix.
 * \return The matrix precision.
 */
bml_matrix_precision_t
bml_get_precision(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case type_uninitialized:
            return precision_uninitialized;
            break;
        case dense:
            return bml_get_precision_dense(A);
            break;
        case ellpack:
            return bml_get_precision_ellpack(A);
            break;
        default:
            LOG_ERROR("unknown precision");
            break;
    }
    return precision_uninitialized;
}

/** Return the matrix size.
 *
 * \param A The matrix.
 * \return The matrix size.
 */
int
bml_get_N(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case type_uninitialized:
            return -1;
            break;
        case dense:
            return bml_get_N_dense(A);
            break;
        case ellpack:
            return bml_get_N_ellpack(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return -1;
}

/** Return the matrix parameter M.
 *
 * \param A The matrix.
 * \return The matrix parameter M.
 */
int
bml_get_M(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case type_uninitialized:
            return 0;
            break;
        case dense:
            return bml_get_M_dense(A);
            break;
        case ellpack:
            return bml_get_M_ellpack(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return -1;
}

/** Return the bandwidth of a row in the matrix.
 *
 * \param A The bml matrix.
 * \param i The row index.
 * \return The bandwidth of row i.
 */
int
bml_get_row_bandwidth(
    const bml_matrix_t * A,
    const int i)
{
    if (i < 0 || i >= bml_get_N(A))
    {
        LOG_ERROR("row index %d outside of matrix\n", i);
    }
    switch (bml_get_type(A))
    {
        case dense:
            return bml_get_row_bandwidth_dense(A, i);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return -1;
}

/** Return the bandwidth of a matrix.
 *
 * \param A The bml matrix.
 * \return The bandwidth of row i.
 */
int
bml_get_bandwidth(
    const bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            return bml_get_bandwidth_dense(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return -1;
}
