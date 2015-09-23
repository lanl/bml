#include "../bml_multiply.h"
#include "../bml_types.h"
#include "bml_add_ellpack.h"
#include "bml_multiply_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/** Matrix multiply.
 *
 * C = alpha * A * B + beta * C
 *
 *  \ingroup multiply_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param C Matrix C
 *  \param alpha Scalar factor multiplied by A * B
 *  \param beta Scalar factor multiplied by C
 *  \param threshold Used for sparse multiply
 */
void bml_multiply_ellpack(const bml_matrix_ellpack_t *A, const bml_matrix_ellpack_t *B, const bml_matrix_ellpack_t *C, const double alpha, const double beta, const double threshold)
{

    double trX = 0.0;
    double trX2 = 0.0;

    switch(A->matrix_precision) {
    case single_real:
//        bml_multiplyX2_ellpack_single(A, C, trX, trX2, threshold);
//        bml_add_ellpack_single(A, C, alpha, beta, threshold);
        break;
    case double_real:
        bml_multiplyX2_ellpack_double(A, C, trX, trX2, threshold);
        bml_add_ellpack_double(A, C, alpha, beta, threshold);
        break;
    }
}

/** Matrix multiply.
 *
 * C = alpha * A * B + beta * C
 *
 *  \ingroup multiply_group
 * 
 *  \param xmatrix Matrix xmatrix 
 *  \param x2matrix Matrix x2matrix
 *  \param trX Trace of xmatrix
 *  \param trX2 Trace of xmatrix^2
 *  \param threshold Used for sparse multiply
 */
void bml_multiplyX2_ellpack_double(const bml_matrix_ellpack_t *xmatrix, const bml_matrix_ellpack_t *x2matrix, double trX, double trX2, const double threshold)
{
  int hsize = xmatrix->N;
  int msize = xmatrix->M;
  int ix[hsize];
  double x[hsize];
  double traceX = 0.0;  
  double traceX2 = 0.0;
  double *xvalue = (double*)xmatrix->value;
  double *x2value = (double*)x2matrix->value;

  memset(ix, 0, hsize*sizeof(int));
  memset(x, 0.0, hsize*sizeof(double));

  #pragma omp parallel for firstprivate(ix,x) reduction(+:traceX,traceX2)
  for(int i = 0; i < hsize; i++)  // CALCULATES THRESHOLDED X^2
  {
    int l = 0;
    for(int jp = 0; jp < xmatrix->nnz[i]; jp++)
    {
      //double a = xmatrix->value[i*msize+jp];
      double a = xvalue[i*msize+jp];
      int j = xmatrix->index[i*msize+jp];
      if (j == i)
      {
        traceX = traceX + a;
      }
      for(int kp = 0; kp < xmatrix->nnz[j]; kp++)
      {
        int k = xmatrix->index[j*msize+kp];
        if (ix[k] == 0)
        {
          x[k] = 0.0;
          x2matrix->index[i*msize+l] = k;
          ix[k] = i+1;
          l++;
        }
        //x[k] = x[k] + a * xmatrix->value[j*msize+kp]; // TEMPORARY STORAGE VECTOR LENGTH FULL N
        x[k] = x[k] + a * xvalue[j*msize+kp]; // TEMPORARY STORAGE VECTOR LENGTH FULL N
      }
    }

    // Check for number of non-zeroes per row exceeded
    if (l > msize)
    {
      printf("\nERROR: Number of non-zeroes per row > M, Increase M\n");
      exit(-1);
    }

    int ll = 0;
    for(int j = 0; j < l; j++)
    {
      int jp = x2matrix->index[i*msize+j];
      double xtmp = x[jp];
      if (jp == i)
      {
        traceX2 = traceX2 + xtmp;
        //x2matrix->value[i*msize+ll] = xtmp;
        x2value[i*msize+ll] = xtmp;
        x2matrix->index[i*msize+ll] = jp;
        ll++;
      }
      else if(fabs(xtmp) > threshold)
      {
        //x2matrix->value[i*msize+ll] = xtmp;
        x2value[i*msize+ll] = xtmp;
        x2matrix->index[i*msize+ll] = jp;
        ll++;
      }
      ix[jp] = 0;
      x[jp] = 0.0;
    }
    x2matrix->nnz[i] = ll;
  }

  trX = traceX;
  trX2 = traceX2;

}
