#include "bml_allocate.h"
#include "bml_copy.h"
#include "bml_introspection.h"
#include "bml_parallel.h"
#include "bml_logger.h"
#include "dense/bml_copy_dense.h"
#include "ellpack/bml_copy_ellpack.h"
#include "ellblock/bml_copy_ellblock.h"
#include "csr/bml_copy_csr.h"
#ifdef BML_USE_MPI
#include "distributed2d/bml_copy_distributed2d.h"
#endif

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/** Copy a matrix - result is a new matrix.
 *
 * \ingroup copy_group_C
 *
 * \param A Matrix to copy
 * \return  A Copy of A
 */
bml_matrix_t *
bml_copy_new(
    bml_matrix_t * A)
{
    bml_matrix_t *B = NULL;

    LOG_DEBUG("creating and copying matrix\n");
    switch (bml_get_type(A))
    {
        case dense:
            B = bml_copy_dense_new(A);
            break;
        case ellpack:
            B = bml_copy_ellpack_new(A);
            break;
        case ellblock:
            B = bml_copy_ellblock_new(A);
            break;
        case csr:
            B = bml_copy_csr_new(A);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            B = bml_copy_distributed2d_new(A);
            break;
#endif
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return B;
}

/** Copy a matrix.
 *
 * \param A Matrix to copy
 * \param B Copy of Matrix A
 */
void
bml_copy(
    bml_matrix_t * A,
    bml_matrix_t * B)
{
    assert(A != NULL);
    assert(B != NULL);
    LOG_DEBUG("copying matrix\n");

    if (bml_get_type(A) != bml_get_type(B))
    {
        LOG_ERROR("type mismatch\n");
    }
    if (bml_get_N(A) < 0)
    {
        LOG_ERROR("matrix is not initialized\n");
    }
    if (bml_get_N(A) != bml_get_N(B))
    {
        LOG_ERROR("matrix size mismatch\n");
    }
    if (bml_get_M(A) > bml_get_M(B) && bml_get_type(A) != csr)
    {
        LOG_ERROR("matrix parameter mismatch\n");
    }
    switch (bml_get_type(A))
    {
        case dense:
            bml_copy_dense(A, B);
            break;
        case ellpack:
            bml_copy_ellpack(A, B);
            break;
        case ellblock:
            bml_copy_ellblock(A, B);
            break;
        case csr:
            bml_copy_csr(A, B);
            break;
#ifdef BML_USE_MPI
        case distributed2d:
            bml_copy_distributed2d(A, B);
            break;
#endif
        default:
            LOG_ERROR("bml_copy --- unknown matrix type\n");
            break;
    }
}

/** Reorder a matrix in place.
 *
 * \ingroup copy_group_C
 *
 * \param A Matrix to reorder
 * \param perm permutation vector for reordering
 */
void
bml_reorder(
    bml_matrix_t * A,
    int *perm)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_reorder_dense(A, perm);
            break;
        case ellpack:
            bml_reorder_ellpack(A, perm);
            break;
        case ellblock:
            bml_reorder_ellblock(A, perm);
            break;
        case csr:
            bml_reorder_csr(A, perm);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Save current domain for bml matrix.
 *
 * \param A Matrix with domain
 */
void
bml_save_domain(
    bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_save_domain_dense(A);
            break;
        case ellpack:
            bml_save_domain_ellpack(A);
            break;
        case csr:
            bml_save_domain_csr(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Restore to saved domain for bml matrix.
 *
 * \param A Matrix with domain
 */
void
bml_restore_domain(
    bml_matrix_t * A)
{
    switch (bml_get_type(A))
    {
        case dense:
            bml_restore_domain_dense(A);
            break;
        case ellpack:
            bml_restore_domain_ellpack(A);
            break;
        case csr:
            bml_restore_domain_csr(A);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}
