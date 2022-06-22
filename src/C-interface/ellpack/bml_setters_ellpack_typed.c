#include "../../macros.h"
#include "../../typed.h"
#include "../bml_introspection.h"
#include "../bml_types.h"
#include "bml_setters_ellpack.h"
#include "bml_types_ellpack.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/** Set element i,j asuming there's no resetting of any element of A.
 *
 *  \ingroup setters
 *
 *  \param A The matrix which takes row i
 *  \param i The column index
 *  \param j The row index
 *  \param value The element to be added
 *  \WARNING sets an element from scratch
 *  \todo set element new.
 *
 *
 */
void TYPED_FUNC(
    bml_set_element_new_ellpack) (
    bml_matrix_ellpack_t * A,
    int i,
    int j,
    void *element)
{
    int A_N = A->N;
    int A_M = A->M;
    int l;
    int ll;

    REAL_T *A_value = (REAL_T *) A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;

#ifdef USE_OMP_OFFLOAD
//#pragma omp target
#pragma omp target update from(A_nnz[:A_N], A_index[:A_N*A_M], A_value[:A_N*A_M])
#endif
    A_value[ROWMAJOR(i, A_nnz[i], A_N, A_M)] = *((REAL_T *) element);
    A_index[ROWMAJOR(i, A_nnz[i], A_N, A_M)] = j;
    A_nnz[i]++;
#ifdef USE_OMP_OFFLOAD
#pragma omp target update to(A_nnz[:A_N], A_index[:A_N*A_M], A_value[:A_N*A_M])
#endif

}


/** Set element i,j of matrix A.
 *
 *  \ingroup setters
 *
 *  \param A The matrix which takes row i
 *  \param i The column index
 *  \param j The row index
 *  \param value The element to be set
 *  \WARNING sets an element from scratch
 *  \todo set element new.
 *
 *
 */
void TYPED_FUNC(
    bml_set_element_ellpack) (
    bml_matrix_ellpack_t * A,
    int i,
    int j,
    void *element)
{
    int A_N = A->N;
    int A_M = A->M;
    int l;
    int ll;

    REAL_T *A_value = (REAL_T *) A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;

#ifdef USE_OMP_OFFLOAD
#pragma omp target update from(A_nnz[:A_N], A_index[:A_N*A_M], A_value[:A_N*A_M])
#endif
    ll = 0;
    if (A_nnz[i] > 2)
    {
        for (int l = 0; l < A_nnz[i]; l++)
        {
            if (A_index[ROWMAJOR(i, l, A_N, A_M)] == j) //Something in the row at the same position
            {
                A_value[ROWMAJOR(i, l, A_N, A_M)] = *((REAL_T *) element);
                ll = 1;
                break;
            }
        }
        if (ll == 0)            //There is something in the row but in a different position
        {
            A_value[ROWMAJOR(i, A_nnz[i], A_N, A_M)] = *((REAL_T *) element);
            A_index[ROWMAJOR(i, A_nnz[i], A_N, A_M)] = j;
            A_nnz[i]++;
        }
    }
    else                        //There is nothing in the row
    {
        A_value[ROWMAJOR(i, A_nnz[i], A_N, A_M)] = *((REAL_T *) element);
        A_index[ROWMAJOR(i, A_nnz[i], A_N, A_M)] = j;
        A_nnz[i]++;
    }
#ifdef USE_OMP_OFFLOAD
#pragma omp target update to(A_nnz[:A_N], A_index[:A_N*A_M], A_value[:A_N*A_M])
#endif
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
    int i,
    void *_row,
    double threshold)
{
    REAL_T *row = _row;
    int A_N = A->N;
    int A_M = A->M;

    int ll = -1;

    REAL_T *A_value = (REAL_T *) A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;

#ifdef USE_OMP_OFFLOAD
#pragma omp target map(to:row[0:A_N])
#endif
    {                           // begin offload region
        for (int j = 0; j < A_N; j++)
        {

            if (ABS(row[j]) > threshold)
            {
                ll++;

                A_value[ROWMAJOR(i, ll, A_N, A_M)] = row[j];
                A_index[ROWMAJOR(i, ll, A_N, A_M)] = j;

            }
        }
        A_nnz[i] = ll + 1;
    }                           // end offload region

}

/** Set diagonal of matrix A.
 *
 *  \ingroup setters
 *
 *  \param A The matrix which takes diag
 *  \param diag The diagonal to be set
 *  \param threshold The threshold value to be used
 */
void TYPED_FUNC(
    bml_set_diagonal_ellpack) (
    bml_matrix_ellpack_t * A,
    void *_diagonal,
    double threshold)
{
    REAL_T *diagonal = _diagonal;
    int A_N = A->N;
    int A_M = A->M;

    REAL_T *A_value = (REAL_T *) A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;

#ifdef USE_OMP_OFFLOAD
#pragma omp target parallel for map(to:diagonal[:A_N])
#endif
    for (int i = 0; i < A_N; i++)
    {
        int ll = 0;
        for (int j = 0; j < A_nnz[i]; j++)
        {
            if (A_index[ROWMAJOR(i, j, A_N, A_M)] == i)
            {
                if (ABS(diagonal[i]) > threshold)
                {
                    A_value[ROWMAJOR(i, j, A_N, A_M)] = diagonal[i];
                }
                else
                {
                    A_value[ROWMAJOR(i, j, A_N, A_M)] = 0.0;
                }
                ll = 1;
            }
        }

        /* If there is no diagonal elements then
         */
        if (ll == 0)
        {
            if (ABS(diagonal[i]) > threshold)
            {
                A_index[ROWMAJOR(i, A_nnz[i], A_N, A_M)] = i;
                A_value[ROWMAJOR(i, A_nnz[i], A_N, A_M)] = diagonal[i];
                A_nnz[i]++;
            }
        }

    }
}
