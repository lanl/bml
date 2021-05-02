#include "../../macros.h"
#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "../bml_logger.h"
#include "../bml_types.h"
#include "bml_introspection_distributed2d.h"
#include "bml_types_distributed2d.h"


/** Return the matrix precision.
 *
 * \param A The matrix.
 * \return The matrix precision.
 */
bml_matrix_precision_t
bml_get_precision_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    if (A != NULL)
    {
        return bml_get_precision(A->matrix);
    }
    else
    {
        return precision_uninitialized;
    }
}

/** Return the matrix distribution mode.
 *
 * \param A The matrix.
 * \return The matrix distribution mode.
 */
bml_distribution_mode_t
bml_get_distribution_mode_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    if (A != NULL)
    {
        return distributed;
    }
    else
    {
        return -1;
    }
}

/** Return the matrix size.
 *
 * \param A The matrix.
 * \return The matrix size.
 */
int
bml_get_N_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    LOG_DEBUG("bml_get_N_distributed2d\n");
    if (A != NULL)
    {
        return A->N;
    }
    else
    {
        return -1;
    }
}

/** Return the matrix parameter M.
 *
 * \param A The matrix.
 * \return The matrix parameter M.
 */
int
bml_get_M_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    if (A != NULL)
    {
        return A->M;
    }
    else
    {
        return -1;
    }
}

bml_matrix_t *
bml_get_local_matrix_distributed2d(
    bml_matrix_distributed2d_t * A)
{
    if (A != NULL)
    {
        return A->matrix;
    }
    else
    {
        return NULL;
    }
}
