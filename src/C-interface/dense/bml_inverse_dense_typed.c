#ifdef BML_USE_MAGMA
#include "magma_v2.h"
//#elif (MKL_GPU)
//#include "stdio.h"
//#include "mkl.h"
//#include "mkl_omp_offload.h"
#else
#include "../blas.h"
#include "../lapack.h"
#endif

#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_copy.h"
#include "../bml_inverse.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_copy_dense.h"
#include "bml_inverse_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Matrix inverse.
 *
 * \f$ B \leftarrow A^-1 f$
 *
 * \ingroup inverse_group
 *
 * \param A Matrix A
 * \param B Inverse of Matrix A
 */
bml_matrix_dense_t *TYPED_FUNC(
    bml_inverse_dense) (
    bml_matrix_dense_t * A)
{
    bml_matrix_dense_t *B = TYPED_FUNC(bml_copy_dense_new) (A);

    TYPED_FUNC(bml_inverse_inplace_dense) (B);

    return B;
}

/** Matrix inverse inplace.
 *
 * \f$ A \leftarrow A^-1 f$
 *
 * \ingroup inverse_group
 *
 * \param A Matrix A
 */
void TYPED_FUNC(
    bml_inverse_inplace_dense) (
    bml_matrix_dense_t * A)
{
    int info;
    int M = A->N;
    int N = A->N;
    int lda = A->ld;
    int *ipiv = bml_allocate_memory(N * sizeof(int));

#ifdef BML_USE_MAGMA
    MAGMAGPU(getrf) (M, N, A->matrix, A->ld, ipiv, &info);
    if (info != 0)
        LOG_ERROR("ERROR in getrf_gpu");
    int nb = magma_get_dgetri_nb(N);
    int lwork = N * nb;
    MAGMA_T *dwork;
    MAGMA(malloc) (&dwork, lwork * sizeof(MAGMA_T));
    MAGMAGPU(getri) (N, A->matrix, A->ld, ipiv, dwork, lwork, &info);
    if (info != 0)
        LOG_ERROR("ERROR in getri_gpu");
    magma_free(dwork);
//#elif defined(MKL_GPU)
    //printf("Got to here 1 \n");

// for now pull the data back to the CPU
    //int sizea = A->N * A->N;
    //int dnum = 0;
//
    //MKL_T *A_matrix = (MKL_T *) A->matrix;
    //MKL_T *wmatrix = (MKL_T *) malloc(sizea*sizeof(REAL_T));
//
    //MKL_INT *la_info = (MKL_INT *) malloc(sizeof(MKL_INT));
    //MKL_INT *g_ipiv = (MKL_INT *) malloc(N*sizeof(MKL_INT));

//#pragma omp target data map(g_ipiv[0:N], wmatrix[0:sizea], la_info[0:1])
//#pragma omp target variant dispatch device(dnum) use_device_ptr(A_matrix, g_ipiv, la_info)
    // la_info = G_LAPACK(getrf) (LAPACK_ROW_MAJOR, M, N, A_matrix, lda, g_ipiv);
    //G_LAPACK(getrf) (&M, &N, A_matrix, &lda, g_ipiv, la_info);
//#pragma omp target update from(A_matrix[0:sizea], g_ipiv[0:N])
    //printf("After getrf \n");
    //bml_print_bml_matrix(A, 0, N, 0, N);
    //for (int i=0; i<N; i++) {
    //    printf("%d,", g_ipiv[i]);
    //}
    //printf("\n");
    //printf("%d \n", la_info);
//#pragma omp target variant dispatch device(dnum) use_device_ptr(A_matrix, g_ipiv, wmatrix, la_info)
    //la_info = G_LAPACK(getri) (LAPACK_ROW_MAJOR, N, A_matrix, lda, g_ipiv);
    //G_LAPACK(getri) (&N, A_matrix, &lda, g_ipiv, wmatrix, &N, la_info);
//#pragma omp target update from(A_matrix[0:sizea])
    //printf("After getri \n");
    //bml_print_bml_matrix(A, 0, N, 0, N);
    //printf("%d \n", la_info);
#else
    int lwork = N * N;
    REAL_T *work = bml_allocate_memory(lwork * sizeof(REAL_T));

#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
#ifdef MKL_GPU
// pull from GPU
    REAL_T *A_matrix = A->matrix;
#pragma omp target update from(A_matrix[0:N*N])
#endif
    C_BLAS(GETRF) (&M, &N, A->matrix, &lda, ipiv, &info);
    C_BLAS(GETRI) (&N, A->matrix, &N, ipiv, work, &lwork, &info);
#ifdef MKL_GPU
// push back to GPU
#pragma omp target update to(A_matrix[0:N*N])
#endif
#endif
    bml_free_memory(work);
#endif /* BML_USE_MAGMA */
    bml_free_memory(ipiv);
}
