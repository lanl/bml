#include "../macros.h"
#include "../typed.h"
#include "../bml_introspection.h"
#include "bml_setters_ellpack.h"
#include "bml_types_ellpack.h"
#include <stdio.h>
#include <complex.h>
#include <stdlib.h>

/** Set row i of matrix A.
 *
 *  \ingroup setters
 *
 *  \param A The matrix which takes row i
 *  \param i The index of the row to be set
 *  \param row The row to be set
 *  \param threshold The row to be set
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

    int ll=0;

    REAL_T *A_value = (REAL_T *) A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;
    
    printf("HOLA %d\n",A_N);
    printf("HOLA %d\n",A_M);
    printf("HOLA %d\n",i);
    printf("HOLA %f\n",threshold);        
   
    for (int j = 0; j < A_N; j++)
    {
      printf("%f \n",row[j]);
        if (ABS(row[j]) > threshold)
	{
	    ll++;
            printf("%d \n",ll);
            A_value[ROWMAJOR(i, ll, A_N, A_M)] = row[j];
	    A_index[ROWMAJOR(i, ll, A_N, A_M)] = j;
	}        
    }
    A_nnz[i] = ll;            
}
