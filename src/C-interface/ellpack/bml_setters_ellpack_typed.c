#include "../macros.h"
#include "../typed.h"
#include "../bml_introspection.h"
#include "bml_setters_ellpack.h"
#include "bml_types_ellpack.h"
#include "bml_types.h" 
#include <stdio.h> 
#include <stdlib.h>  
#include <complex.h>
#include <math.h> 

void TYPED_FUNC(
    bml_set_ellpack) (
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value)
{
}

/** Set row i of matrix A.
  *
  *  \ingroup setters
  *
  *  \param A The matrix which takes row i
  *  \param i The index of the row to be set
  *  \param row The row to be set
  *  \param threshold The threshold value to be set
  *
  */

void TYPED_FUNC(
    bml_set_row_ellpack) (
      bml_matrix_ellpack_t * A,
      const int i,
      const REAL_T * row,
      const double threshold)
{
  int A_N = A->N;
  int A_M = A->M;

  int ll = 0;

  REAL_T *A_value = (REAL_T *) A->value;
  int *A_index = A->index;
  int *A_nnz = A->nnz;

  for (int j = 0; j < A_M; j++)
  { 
    A_value[ROWMAJOR(i, j, A_N, A_M)] = 0.0; /* set all the previous values to 0 */

    if (ABS(row[j]) > threshold)
    {
      ll++;
      A_value[ROWMAJOR(i, ll, A_N, A_M)] = row[j];
      A_index[ROWMAJOR(i, ll, A_N, A_M)] = j;
    }
  }
  A_nnz[i] = ll;
}


/** Set diagonal of matrix A.
  *
  *  \ingroup setters
  *
  *  \param A The matrix which takes diag
  *  \param diag The diagonal to be set
  *  \param threshold The threshold value to be used
  *
  */
 
void TYPED_FUNC(
    bml_set_diag_ellpack) (
      bml_matrix_ellpack_t * A,
      const REAL_T * diag,
      const double threshold)
{
  int A_N = A->N;
  int A_M = A->M;

  REAL_T *A_value = (REAL_T *) A->value;
  int *A_index = A->index;
  int *A_nnz = A->nnz;

  for (int i = 0; i < A_N; i++)
  {
    for (int j = 0; j < A_M; j++) 
    {
      if (A_index[ROWMAJOR(i, j, A_N, A_M)] == i)
      {
        if (ABS(diag[i]) > threshold) {
          A_value[ROWMAJOR(i, j, A_N, A_M)] = diag[i];
        }
        else {
          A_value[ROWMAJOR(i, j, A_N, A_M)] = 0.0; 
        }
      }
    }
    if (A_nnz[i] < 1) /* If there is nothing in the row the nnz will have to be at least 1 */
    {
      A_nnz[i] = 1;
    }
  }
}
