#include "../macros.h"
#include "../typed.h"
#include "../bml_logger.h"
#include "../bml_introspection.h"
#include "bml_getters_ellpack.h"
#include "bml_types_ellpack.h"

#include <complex.h>


/** Get the diagonal of matrix A.
 * 
 *  \ingroup getters
 *
 *  \param A The matrix which takes row i
 *  \param Diagonal Array to copy the diagonal
 *
 */

void TYPED_FUNC(
  bml_get_diagonal_ellpack) (
    bml_matrix_ellpack_t * A,
    REAL_T * diagonal)
  {    
    int A_N = A->N;
    int A_M = A->M;  
    REAL_T *A_value = (REAL_T *) A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;
    
    for (int i = 0; i < A_N; i++)
    {
      diagonal[i] = 0.0;
      for (int j = 0; j < A_nnz[i]; j++)
      {
        if(A_index[ROWMAJOR(i, j, A_N, A_M)] == i)
        {
          diagonal[i] = A_value[ROWMAJOR(i, j, A_N, A_M)];
        }
      }
    }
  }
  

/** Get row i of matrix A.
 *
 *  \ingroup getters
 *
 *  \param A The matrix which takes row i
 *  \param i The index of the row to get
 *  \param row Array to copy the row
 *
 */

void TYPED_FUNC(
    bml_get_row_ellpack) (
    bml_matrix_ellpack_t * A,
    const int i,
    REAL_T * row)
{

    int ll;
    int A_N = A->N;
    int A_M = A->M;
    REAL_T *A_value = (REAL_T *) A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;

    for (int i = 0; i < A_N; i++)
    {
        row[i] = 0.0;
    }

    for (int j = 0; j < A_nnz[i]; j++)
    {
        ll = A_index[ROWMAJOR(i, j, A_N, A_M)];
        if (ll >= 0)
        {
            row[ll] = A_value[ROWMAJOR(i, j, A_N, A_M)];
        }
    }
}
