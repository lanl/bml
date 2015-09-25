#include "../typed.h"
#include "bml_allocate.h"
#include "bml_add.h"
#include "bml_types.h"
#include "bml_allocate_ellpack.h"
#include "bml_add_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

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
void TYPED_FUNC(bml_add_ellpack) (const bml_matrix_ellpack_t *A, const bml_matrix_ellpack_t *B, const double alpha, const double beta, const double threshold)
{
    REAL_T salpha = (REAL_T)alpha;
    REAL_T sbeta = (REAL_T)beta;
    REAL_T sthreshold = (REAL_T)threshold;

    int hsize = A->N;
    int msize = A->M;
    int ix[hsize];

    REAL_T x[hsize];
    REAL_T *A_value = (REAL_T*)A->value;
    REAL_T *B_value = (REAL_T*)B->value;

    memset(ix, 0, hsize*sizeof(int));
    memset(x, 0.0, hsize*sizeof(REAL_T));

  #pragma omp parallel for firstprivate(x,ix)
  for (int i = 0; i < hsize; i++) 
  {
    int l = 0;
    for(int jp = 0; jp < A->nnz[i]; jp++)
    {
      int k = A->index[i*msize+jp];
      if (ix[k] == 0)
      {
        x[k] = 0.0;
        ix[k] = i+1;
        A->index[i*msize+l] = k;
        l++;
      }
      x[k] = x[k] + salpha * A_value[i*msize+jp];
    }

    for(int jp = 0; jp < B->nnz[i]; jp++)
    {
      int k = B->index[i*msize+jp];
      if (ix[k] == 0)
      {
        x[k] = 0.0;
        ix[k] = i+1;
        A->index[i*msize+l] = k;
        l++;
      }
      x[k] = x[k] + sbeta * B_value[i*msize+jp];
    }
    A->nnz[i] = l;

    int ll = 0;
    for(int jp = 0; jp < l; jp++)
    {
      REAL_T xTmp = x[A->index[i*msize+jp]];
      if (fabs(xTmp) > sthreshold) // THIS THRESHOLDING COULD BE IGNORED!?
      {
        A_value[i*msize+ll] = xTmp;
        A->index[i*msize+ll] = A->index[i*msize+jp];
        ll++;
      }
      x[A->index[i*msize+jp]] = 0.0;
      ix[A->index[i*msize+jp]] = 0;
    }
    A->nnz[i] = ll;
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
void TYPED_FUNC(bml_add_identity_ellpack) (const bml_matrix_ellpack_t *A, const double beta, const double threshold)
{
    REAL_T alpha = (REAL_T)1.0; 

    bml_matrix_ellpack_t *I = TYPED_FUNC(bml_identity_matrix_ellpack)(A->N, A->M);
    
    TYPED_FUNC(bml_add_ellpack)(A, I, alpha, beta, threshold);
}
