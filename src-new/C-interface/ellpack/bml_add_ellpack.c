#include "../bml_allocate.h"
#include "../bml_add.h"
#include "../bml_types.h"
#include "bml_allocate_ellpack.h"
#include "bml_add_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/** Matrix addition.
 *
 * A = alpha * A + beta * B
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param alpha Scalar factor multiplied by A
 *  \param beta Scalar factor multiplied by B
 *  \param threshold Threshold for matrix addition
 */
void bml_add_ellpack(const bml_matrix_ellpack_t *A, const bml_matrix_ellpack_t *B, const double alpha, const double beta, const double threshold)
{
    float alpha_s, beta_s;

    int nElems = B->N * B->M;
    int inc = 1;
    
    switch(B->matrix_precision) {
    case single_real:
        alpha_s = (float)alpha;
        beta_s = (float)beta;

        break;
    case double_real:
        bml_add_ellpack_double(A, B, alpha, beta, threshold);
        break;
    }
}

/** Matrix addition.
 *
 *  A = A + beta * I
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param beta Scalar factor multiplied by A
 *  \param threshold Threshold for matrix addition
 */
void bml_add_identity_ellpack(const bml_matrix_ellpack_t *A, const double beta, const double threshold)
{
    float alpha_s, beta_s;
    double alpha;

    int nElems = A->N * A->M;
    int inc = 1;

    bml_matrix_ellpack_t *I = bml_identity_matrix_ellpack(A->matrix_precision, A->N, A->M);
    
    switch(A->matrix_precision) {
    case single_real:
        alpha_s = (float)1.0;
        beta_s = (float)beta;
//        bml_add_ellpack_single(A, I, alpha_s, beta_s, threshold);
        break;
    case double_real:
        alpha = (double)1.0;
        bml_add_ellpack_double(A, I, alpha, beta, threshold);
        break;
    }
}

/** Sparse Matrix addition.
 *
 *  A = alpha * A + beta * B
 *
 *  \ingroup add_group
 *
 *  \param xmatrix Matrix xmatrix 
 *  \param x2matrix Matrix x2matrix 
 *  \param alpha Scalar factor multiplied by x2matrix 
 *  \param beta Scalar factor multiplied by x2matrix 
 *  \param threshold Threshold for matrix addition
 */
void bml_add_ellpack_double(const bml_matrix_ellpack_t *xmatrix, const bml_matrix_ellpack_t *x2matrix, const double alpha, const double beta, const double threshold)
{
  int hsize = xmatrix->N;
  int msize = xmatrix->M;
  int ix[hsize];
  double x[hsize];
  double *xvalue = (double*)xmatrix->value;
  double *x2value = (double*)x2matrix->value;

  memset(ix, 0, hsize*sizeof(int));
  memset(x, 0.0, hsize*sizeof(double));

  #pragma omp parallel for firstprivate(x,ix)
  for (int i = 0; i < hsize; i++) // X = 2X-X^2
  {
    int l = 0;
    for(int jp = 0; jp < xmatrix->nnz[i]; jp++)
    {
      int k = xmatrix->index[i*msize+jp];
      if (ix[k] == 0)
      {
        x[k] = 0.0;
        ix[k] = i+1;
        xmatrix->index[i*msize+l] = k;
        l++;
      }
      //x[k] = x[k] + alpha * xmatrix->value[i*msize+jp];
      x[k] = x[k] + alpha * xvalue[i*msize+jp];
    }

    for(int jp = 0; jp < x2matrix->nnz[i]; jp++)
    {
      int k = x2matrix->index[i*msize+jp];
      if (ix[k] == 0)
      {
        x[k] = 0.0;
        ix[k] = i+1;
        xmatrix->index[i*msize+l] = k;
        l++;
      }
      //x[k] = x[k] + beta * x2matrix->value[i*msize+jp];
      x[k] = x[k] + beta * x2value[i*msize+jp];
    }
    xmatrix->nnz[i] = l;

    int ll = 0;
    for(int jp = 0; jp < l; jp++)
    {
      double xTmp = x[xmatrix->index[i*msize+jp]];
      if (fabs(xTmp) > threshold) // THIS THRESHOLDING COULD BE IGNORED!?
      {
        //xmatrix->value[i*msize+ll] = xTmp;
        xvalue[i*msize+ll] = xTmp;
        xmatrix->index[i*msize+ll] = xmatrix->index[i*msize+jp];
        ll++;
      }
      x[xmatrix->index[i*msize+jp]] = 0.0;
      ix[xmatrix->index[i*msize+jp]] = 0;
    }
    xmatrix->nnz[i] = ll;
  }
}
