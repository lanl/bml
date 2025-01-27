#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "bml_setters_dense.h"
#include "bml_types_dense.h"

#ifdef BML_USE_MAGMA
#include "magma_v2.h"
#endif

void bml_set_N_dense(
    bml_matrix_dense_t * A,
    int N)
{
  A->N = N;
#ifdef BML_USE_MAGMA
  A->ld = magma_roundup(A->N, 32);
#else
  A->ld = A->N;
#endif
}

void
bml_set_element_dense(
    bml_matrix_dense_t * A,
    int i,
    int j,
    void *value)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_element_dense_single_real(A, i, j, value);
            break;
        case double_real:
            bml_set_element_dense_double_real(A, i, j, value);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_set_element_dense_single_complex(A, i, j, value);
            break;
        case double_complex:
            bml_set_element_dense_double_complex(A, i, j, value);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision for bml_set_element_dense\n");
            break;
    }
}

void
bml_set_row_dense(
    bml_matrix_dense_t * A,
    int i,
    void *row)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_row_dense_single_real(A, i, row);
            break;
        case double_real:
            bml_set_row_dense_double_real(A, i, row);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_set_row_dense_single_complex(A, i, row);
            break;
        case double_complex:
            bml_set_row_dense_double_complex(A, i, row);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}

/** Setters for diagonal.
 *
 * \param A The matrix.
 * \param diag The diagonal (a vector with all diagonal elements).
 */
void
bml_set_diagonal_dense(
    bml_matrix_dense_t * A,
    void *diagonal)
{
    switch (bml_get_precision(A))
    {
        case single_real:
            bml_set_diagonal_dense_single_real(A, diagonal);
            break;
        case double_real:
            bml_set_diagonal_dense_double_real(A, diagonal);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_set_diagonal_dense_single_complex(A, diagonal);
            break;
        case double_complex:
            bml_set_diagonal_dense_double_complex(A, diagonal);
            break;
#endif
        default:
            LOG_ERROR("unkonwn precision\n");
            break;
    }
}
