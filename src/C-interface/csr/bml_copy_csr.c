#include "../bml_copy.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "../bml_domain.h"
#include "bml_types_csr.h"
#include "bml_copy_csr.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/** Copy an csr matrix - result is a new matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_csr_t *
bml_copy_csr_new(
    bml_matrix_csr_t * A)
{
    bml_matrix_csr_t *B = NULL;

    switch (A->matrix_precision)
    {
        case single_real:
            B = bml_copy_csr_new_single_real(A);
            break;
        case double_real:
            B = bml_copy_csr_new_double_real(A);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            B = bml_copy_csr_new_single_complex(A);
            break;
        case double_complex:
            B = bml_copy_csr_new_double_complex(A);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return B;
}

/** Copy an csr matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \param B Copy of matrix A
 */
void
bml_copy_csr(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B)
{

    switch (A->matrix_precision)
    {
        case single_real:
            bml_copy_csr_single_real(A, B);
            break;
        case double_real:
            bml_copy_csr_double_real(A, B);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_copy_csr_single_complex(A, B);
            break;
        case double_complex:
            bml_copy_csr_double_complex(A, B);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

/** Reorder an csr matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be reordered
 *  \param B The permutation matrix
 */
void
bml_reorder_csr(
    bml_matrix_csr_t * A,
    int *perm)
{

    switch (A->matrix_precision)
    {
        case single_real:
            bml_reorder_csr_single_real(A, perm);
            break;
        case double_real:
            bml_reorder_csr_double_real(A, perm);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_reorder_csr_single_complex(A, perm);
            break;
        case double_complex:
            bml_reorder_csr_double_complex(A, perm);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

/** Save the domain for a csr matrix.
 *
 * \ingroup copy_group
 *
 * \param A The matrix with the domain to be saved
 */
void
bml_save_domain_csr(
    bml_matrix_csr_t * A)
{
    LOG_ERROR("bml_save_domain_csr not implemented");
}

/** Restore the domain for a csr matrix.
 *
 * \ingroup copy_group
 *
 * \param A The matrix with the domain to be restored
 */
void
bml_restore_domain_csr(
    bml_matrix_csr_t * A)
{
    LOG_ERROR("bml_restore_domain_csr not implemented");
}
