#include "../../macros.h"
#include "../bml_logger.h"
#include "../bml_submatrix.h"
#include "../bml_types.h"
#include "bml_submatrix_csr.h"
#include "bml_types_csr.h"

#include <stdlib.h>

/** Extract submatrix into new matrix of same format
 *
 * \ingroup submatrix_group_C
 *
 * \param A Matrix A to extract submatrix from
 * \param irow Index of first row to extract
 * \param icol Index of first column to extract
 * \param B_N Number of rows/columns to extract
 * \param B_M Max number of non-zero elemnts/row in exttacted matrix
 */
bml_matrix_csr_t *
bml_extract_submatrix_csr(
    bml_matrix_csr_t * A,
    int irow,
    int icol,
    int B_N,
    int B_M)
{
    switch (A->matrix_precision)
    {
        case single_real:
            return bml_extract_submatrix_csr_single_real(A, irow, icol,
                                                         B_N, B_M);
            break;
        case double_real:
            return bml_extract_submatrix_csr_double_real(A, irow, icol,
                                                         B_N, B_M);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            return bml_extract_submatrix_csr_single_complex(A, irow, icol,
                                                            B_N, B_M);
            break;
        case double_complex:
            return bml_extract_submatrix_csr_double_complex(A, irow, icol,
                                                            B_N, B_M);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
    return NULL;
}

void
bml_assign_submatrix_csr(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    int irow,
    int icol)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_assign_submatrix_csr_single_real(A, B, irow, icol);
            break;
        case double_real:
            bml_assign_submatrix_csr_double_real(A, B, irow, icol);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_assign_submatrix_csr_single_complex(A, B, irow, icol);
            break;
        case double_complex:
            bml_assign_submatrix_csr_double_complex(A, B, irow, icol);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
