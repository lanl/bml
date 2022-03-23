#include "../../macros.h"
#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "../bml_logger.h"
#include "../bml_types.h"
#include "../bml_parallel.h"
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

/** Return the sparsity of a matrix.
 *
 *  Note that the the sparsity of a matrix is defined
 *  as NumberOfZeroes/N*N where N is the matrix dimension.
 *  The density of matrix A will be defined as 1-sparsity(A)
 *
 * \ingroup introspection_group_C
 *
 * \param A The bml matrix.
 * \param threshold The threshold used to compute the sparsity.
 * \return The sparsity of A.
 */
double
bml_get_sparsity_distributed2d(
    bml_matrix_distributed2d_t * A,
    double threshold)
{
    double sp = bml_get_sparsity(A->matrix, threshold);
    int nloc = A->N / A->nprows;
    sp *= nloc * nloc;

    bml_sumRealReduce(&sp);
    sp /= (A->N * A->N);

    return sp;
}
