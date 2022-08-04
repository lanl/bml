#include "../../macros.h"
#include "../bml_logger.h"
#include "bml_elemental_dense.h"
#include "bml_types_dense.h"

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
bml_get_element_dense_single_real(
    bml_matrix_dense_t * A,
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
    return ((float *) A->matrix)[ROWMAJOR(i, j, A->N, A->N)];
}

/** Return a single matrix element.
 *
 * \param A The bml matrix
 * \param i The row index
 * \param j The column index
 * \return The matrix element
 */
double
bml_get_element_dense_double_real(
    bml_matrix_dense_t * A,
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
    return ((double *) A->matrix)[ROWMAJOR(i, j, A->N, A->N)];
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
bml_get_element_dense_single_complex(
    bml_matrix_dense_t * A,
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
    return ((float complex *) A->matrix)[ROWMAJOR(i, j, A->N, A->N)];
}

/** Return a single matrix element.
 *
 * \param A The bml matrix
 * \param i The row index
 * \param j The column index
 * \return The matrix element
 */
double _Complex
bml_get_element_dense_double_complex(
    bml_matrix_dense_t * A,
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
    return ((double complex *) A->matrix)[ROWMAJOR(i, j, A->N, A->N)];
}
#endif
