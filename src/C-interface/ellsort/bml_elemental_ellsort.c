#include "../bml_logger.h"
#include "../../macros.h"
#include "bml_elemental_ellsort.h"
#include "bml_types_ellsort.h"

#ifdef BML_COMPLEX
#include <complex.h>
#endif

/** Return a single matrix element.
 *
 * \param A The bml matrix
 * \param i The row index
 * \param j The column index
 * \return The matrix element
 */
float
bml_get_element_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    int i,
    int j)
{
    float *A_value = (float *) A->value;
    if (i < 0 || i >= A->N)
    {
        LOG_ERROR("row index out of bounds\n");
        return -1;
    }
    if (j < 0 || j >= A->N)
    {
        LOG_ERROR("column index out of bounds\n");
        return -1;
    }
    for (int j_index = 0; j_index < A->nnz[i]; j_index++)
    {
        if (A->index[ROWMAJOR(i, j_index, A->N, A->M)] == j)
        {
            return A_value[ROWMAJOR(i, j_index, A->N, A->M)];
        }
    }
    return 0;
}

/** Return a single matrix element.
 *
 * \param A The bml matrix
 * \param i The row index
 * \param j The column index
 * \return The matrix element
 */
double
bml_get_element_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    int i,
    int j)
{
    if (i < 0 || i >= A->N)
    {
        LOG_ERROR("row index out of bounds\n");
        return -1;
    }
    if (j < 0 || j >= A->N)
    {
        LOG_ERROR("column index out of bounds\n");
        return -1;
    }
    for (int j_index = 0; j_index < A->nnz[i]; j_index++)
    {
        if (A->index[j_index] == j)
        {
            return ((double *) A->value)[ROWMAJOR(i, j_index, A->N, A->M)];
        }
    }
    return 0;
}

#ifdef BML_COMPLEX
/** Return a single matrix element.
 *
 * \param A The bml matrix
 * \param i The row index
 * \param j The column index
 * \return The matrix element
 */
float _Complex
bml_get_element_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    int i,
    int j)
{
    if (i < 0 || i >= A->N)
    {
        LOG_ERROR("row index out of bounds\n");
        return -1;
    }
    if (j < 0 || j >= A->N)
    {
        LOG_ERROR("column index out of bounds\n");
        return -1;
    }
    for (int j_index = 0; j_index < A->nnz[i]; j_index++)
    {
        if (A->index[j_index] == j)
        {
            return ((float complex *)
                    A->value)[ROWMAJOR(i, j_index, A->N, A->M)];
        }
    }
    return 0;
}

/** Return a single matrix element.
 *
 * \param A The bml matrix
 * \param i The row index
 * \param j The column index
 * \return The matrix element
 */
double _Complex
bml_get_element_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    int i,
    int j)
{
    if (i < 0 || i >= A->N)
    {
        LOG_ERROR("row index out of bounds\n");
        return -1;
    }
    if (j < 0 || j >= A->N)
    {
        LOG_ERROR("column index out of bounds\n");
        return -1;
    }
    for (int j_index = 0; j_index < A->nnz[i]; j_index++)
    {
        if (A->index[j_index] == j)
        {
            return ((double complex *)
                    A->value)[ROWMAJOR(i, j_index, A->N, A->M)];
        }
    }
    return 0;
}
#endif
