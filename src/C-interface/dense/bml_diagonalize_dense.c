#include "../../macros.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_diagonalize_dense.h"
#include "bml_types_dense.h"
#include "../bml_utilities.h"

#include <float.h>

#ifdef BML_USE_MAGMA
#include "magma_v2.h"
#ifdef BML_USE_CUSOLVER
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif
#ifdef BML_USE_ROCSOLVER
#include <hip/hip_runtime_api.h>
#include <rocblas.h>
#include <rocsolver.h>
#endif
#endif

#ifdef MKL_GPU
#include "stdio.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#else
#include "../lapack.h"
#endif

#include <string.h>
#include <complex.h>

/** \page diagonalize
 *
 * Note: We can't generify these functions easily since the API
 * differs between the real and complex types. rwork and lrwork are
 * only used in the complex cases. We opted instead to explicitly
 * implement the four versions.
 */
#ifdef BML_USE_MAGMA
void
bml_diagonalize_dense_magma_single_real(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
    int info;
    float *A_matrix;
    float *typed_eigenvalues = (float *) eigenvalues;

    int nb = magma_get_ssytrd_nb(A->N);
    float *evecs;
    magma_int_t ret = magma_smalloc(&evecs, A->N * A->ld);
    assert(ret == MAGMA_SUCCESS);

    float *evals = malloc(A->N * sizeof(float));
    int lwork = 2 * A->N + A->N * nb;
    int tmp = 1 + 6 * A->N + 2 * A->N * A->N;
    if (tmp > lwork)
        lwork = tmp;
    float *work = malloc(lwork * sizeof(float));
    int liwork = 3 + 5 * A->N;
    int *iwork = malloc(liwork * sizeof(int));
    int ldwa = A->ld;
    float *wa = malloc(A->N * ldwa * sizeof(float));

    //copy matrix into evecs
    magmablas_slacpy(MagmaFull, A->N, A->N, A->matrix, A->ld, evecs, A->ld,
                     bml_queue());
    magma_queue_sync(bml_queue());

    magma_ssyevd_gpu(MagmaVec, MagmaUpper, A->N, evecs, A->ld, evals,
                     wa, ldwa, work, lwork, iwork, liwork, &info);
    if (info != 0)
        LOG_ERROR("ERROR in magma_ssyevd_gpu");

    A_matrix = (float *) eigenvectors->matrix;

    magmablas_stranspose(A->N, A->N, evecs, A->ld,
                         A_matrix, eigenvectors->ld, bml_queue());
    for (int i = 0; i < A->N; i++)
        typed_eigenvalues[i] = (float) evals[i];
    magma_queue_sync(bml_queue());

    magma_free(evecs);
    free(wa);
    free(evals);
    free(work);
}
#endif

#ifdef MKL_GPU
void
bml_diagonalize_dense_gpu_single_real(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else

    int dnum = 0;
    MKL_INT info;
    MKL_INT N = A->N;
    float *A_matrix = (float *) A->matrix;
    float *typed_eigenvalues = (float *) eigenvalues;

    float *evecs = (float *) malloc(A->N * A->N * sizeof(float));
    float *evals = (float *) malloc(A->N * sizeof(float));

#pragma omp target enter data map(alloc:evecs[0:N*N])
#pragma omp target enter data map(alloc:evals[0:N])
// pull from GPU
#pragma omp target update from(A_matrix[0:N*N])
    printf("Checking A matrix values \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %6.3f", A_matrix[i * N + j]);
        }
        printf("\n");
    }
    // copy A to evecs on GPU
#pragma omp target teams distribute parallel for
    for (int i = 0; i < N * N; i++)
    {
        evecs[i] = A_matrix[i];
    }
#pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++)
    {
        evals[i] = 1.0;
    }

#pragma omp target update from(evecs[0:N*N])
#pragma omp target update from(evals[0:N])
    // Check values before syev
    printf("Checking evecs input values \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %6.3f,", evecs[i * N + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < N; i++)
    {
        printf("%d, %f \n", i, evals[i]);
    }
#ifdef BML_SYEVD
    // Divide and conquer solver
    MKL_INT lwork = 1 + 6 * N + 2 * N * N;
    float *work = (float *) malloc(lwork * sizeof(float));
    MKL_INT liwork = 3 + 5 * N;
    MKL_INT *iwork = (MKL_INT *) malloc(liwork * sizeof(MKL_INT));
#pragma omp target enter data map(alloc:work[0:lwork])
#pragma omp target enter data map(alloc:iwork[0:liwork])

#pragma omp target variant dispatch use_device_ptr(evecs, evals, work, iwork)
    ssyevd("V", "U", &N, evecs, &N, evals, work, &lwork, iwork, &liwork, &info);
    free(iwork);
#else
    MKL_INT lwork = 3 * N;
    float *work = (float *) mkl_malloc(lwork * sizeof(float), 64);
#pragma omp target enter data map(alloc:work[0:lwork])
#pragma omp target variant dispatch device(dnum) use_device_ptr(evecs, evals, work)
    ssyev("V", "U", &N, evecs, &N, evals, work, &lwork, &info);
#endif // BML_SYEVD

#ifdef MKL_GPU_DEBUG
#pragma omp target update from(evecs[0:N*N])
#pragma omp target update from(evals[0:N])

    printf("After SYEV \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %6.3f", evecs[i * N + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < N; i++)
    {
        printf("%d, %f \n", i, evals[i]);
    }
#endif

// need typed_eigenvalues on CPU
#pragma omp target update from(evals[0:N])
    for (int i = 0; i < N; i++)
    {
        typed_eigenvalues[i] = (float) evals[i];
    }

// leave eigenvectors on GPU
    float *e_matrix = (float *) eigenvectors->matrix;
#pragma omp target enter data map(alloc:e_matrix[0:N*N])
#pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            e_matrix[ROWMAJOR(i, j, N, N)] = evecs[COLMAJOR(i, j, N, N)];
        }
    }

// push eigenvectors back from GPU
// #pragma omp target update from(e_matrix[0:N*N])

    free(evecs);
    free(evals);
    free(work);
#endif // NOBLAS
}
#endif // MKL_GPU

void
bml_diagonalize_dense_cpu_single_real(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    int info;
    int N = A->N;
    float *A_matrix;
    float *typed_eigenvalues = (float *) eigenvalues;

    float *evecs = calloc(A->N * A->N, sizeof(float));
    float *evals = calloc(A->N, sizeof(float));

    memcpy(evecs, A->matrix, A->N * A->N * sizeof(float));

#ifdef BML_SYEVD
    // Divide and conquer solver
    const int lwork = 1 + 6 * N + 2 * N * N;
    float *work = malloc(lwork * sizeof(float));
    const int liwork = 3 + 5 * N;
    int *iwork = malloc(liwork * sizeof(int));
    printf("%d, %d \n", lwork, liwork);

    C_SSYEVD("V", "U", &N, evecs, &N, evals, work, &lwork,
             iwork, &liwork, &info);
    free(iwork);
#else
    const int lwork = 3 * A->N;
    float *work = calloc(lwork, sizeof(float));
    C_SSYEV("V", "U", &N, evecs, &N, evals, work, &lwork, &info);
#endif // BML_SYEVD

    A_matrix = (float *) eigenvectors->matrix;
    for (int i = 0; i < A->N; i++)
    {
        typed_eigenvalues[i] = (float) evals[i];
        for (int j = 0; j < A->N; j++)
        {
            A_matrix[ROWMAJOR(i, j, A->N, A->N)] =
                evecs[COLMAJOR(i, j, A->N, A->N)];
        }
    }

    free(evecs);
    free(evals);
    free(work);
#endif // NOBLAS
}

void
bml_diagonalize_dense_single_real(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
#ifdef BML_USE_MAGMA
    bml_diagonalize_dense_magma_single_real(A, eigenvalues, eigenvectors);
#elif MKL_GPU
    bml_diagonalize_dense_gpu_single_real(A, eigenvalues, eigenvectors);
#else
    bml_diagonalize_dense_cpu_single_real(A, eigenvalues, eigenvectors);
#endif
}

#ifdef BML_USE_MAGMA
void
bml_diagonalize_dense_magma_double_real(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
    int info;
    double *A_matrix;
    double *typed_eigenvalues = (double *) eigenvalues;

    //copy matrix into evecs
    double *evecs;
    magma_int_t ret = magma_dmalloc(&evecs, A->N * A->ld);
    assert(ret == MAGMA_SUCCESS);

    magmablas_dlacpy(MagmaFull, A->N, A->N, A->matrix, A->ld, evecs, A->ld,
                     bml_queue());
    magma_queue_sync(bml_queue());

#ifdef BML_USE_CUSOLVER
    // create cusolver/cublas handle
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // allocate memory for eigenvalues
    double *d_W = NULL;
    cudaError_t cudaStat = cudaMalloc((void **) &d_W, sizeof(double) * A->N);
    assert(cudaSuccess == cudaStat);

    // compute eigenvalues and eigenvectors
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // query working space of syevd
    int lwork = 0;
    cusolver_status =
        cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, A->N, evecs, A->ld,
                                    d_W, &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    double *d_work = NULL;
    cudaStat = cudaMalloc((void **) &d_work, sizeof(double) * lwork);
    assert(cudaSuccess == cudaStat);

    // solve
    int *devInfo = NULL;
    cudaStat = cudaMalloc((void **) &devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat);

    cusolver_status =
        cusolverDnDsyevd(cusolverH, jobz, uplo, A->N, evecs, A->ld, d_W,
                         d_work, lwork, devInfo);
    cudaStat = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat);

    // copy eigenvalues to CPU
    cudaStat =
        cudaMemcpy(typed_eigenvalues, d_W, sizeof(double) * A->N,
                   cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat);

    // free resources
    cudaFree(d_W);
    cudaFree(devInfo);
    cudaFree(d_work);

    if (cusolverH)
        cusolverDnDestroy(cusolverH);
#else // BML_USE_CUSOLVER
#ifdef BML_USE_ROCSOLVER
    // See https://rocsolver.readthedocs.io/_/downloads/en/latest/pdf/
    // create cusolver/cublas handle
    rocblas_handle rocblasH = NULL;
    rocblas_status rocblasS = rocblas_create_handle(&rocblasH);
    assert(rocblas_status_success == rocblasS);

    // allocate memory for eigenvalues
    double *d_W = NULL;
    hipError_t hipStat = hipMalloc((void **) &d_W, sizeof(double) * A->N);
    assert(hipSuccess == hipStat);

    // compute eigenvalues and eigenvectors
    rocblas_evect evect = rocblas_evect_original;
    rocblas_fill uplo = rocblas_fill_lower;

    // allocate working space of syevd
    double *d_work = NULL;
    hipStat = hipMalloc((void **) &d_work, sizeof(double) * A->N * A->N);
    assert(hipSuccess == hipStat);

    // solve
    rocblas_int *devInfo = NULL;
    hipStat = hipMalloc((void **) &devInfo, sizeof(rocblas_int));
    assert(hipSuccess == hipStat);

    rocblasS =
        rocsolver_dsyevd(rocblasH, evect, uplo, A->N, evecs, A->ld, d_W,
                         d_work, devInfo);
    hipStat = hipDeviceSynchronize();
    assert(rocblas_status_success == rocblasS);
    assert(hipSuccess == hipStat);

    // copy eigenvalues to CPU
    hipStat =
        hipMemcpy(typed_eigenvalues, d_W, sizeof(double) * A->N,
                  hipMemcpyDeviceToHost);
    assert(hipSuccess == hipStat);

    // free resources
    hipFree(d_W);
    hipFree(devInfo);
    hipFree(d_work);

    if (rocblasH)
        rocblas_destroy_handle(rocblasH);

#else // MAGMA
    int nb = magma_get_ssytrd_nb(A->N);

    double *evals = malloc(A->N * sizeof(double));
    int lwork = 2 * A->N + A->N * nb;
    int tmp = 1 + 6 * A->N + 2 * A->N * A->N;
    if (tmp > lwork)
        lwork = tmp;
    double *work = malloc(lwork * sizeof(double));
    int liwork = 3 + 5 * A->N;
    int *iwork = malloc(liwork * sizeof(int));
    int ldwa = A->N;
    double *wa = malloc(A->N * ldwa * sizeof(double));

    magma_dsyevd_gpu(MagmaVec, MagmaUpper, A->N, evecs, A->ld, evals,
                     wa, ldwa, work, lwork, iwork, liwork, &info);
    if (info != 0)
        LOG_ERROR("ERROR in magma_dsyevd_gpu");

    free(wa);
    free(work);

    for (int i = 0; i < A->N; i++)
        typed_eigenvalues[i] = (double) evals[i];
    free(evals);

    //verify norm eigenvectors
    //for(int i= 0;i<A->N;i++)
    //{
    //    double norm = magma_dnrm2(A->N, evecs+A->ld*i, 1, bml_queue());
    //    printf("norm = %le\n", norm);
    //}
#endif
#endif // BML_USE_CUSOLVER
    // transpose eigenvactors matrix on GPU
    A_matrix = (double *) eigenvectors->matrix;
    magmablas_dtranspose(A->N, A->N, evecs, A->ld,
                         A_matrix, eigenvectors->ld, bml_queue());
    magma_queue_sync(bml_queue());

    //verify norm eigenvectors transposed
    //for(int i= 0;i<A->N;i++)
    //{
    //    double norm = magma_dnrm2(A->N, evecs+i, A->ld, bml_queue());
    //    printf("norm transposed vector = %le\n", norm);
    //}
    magma_free(evecs);
}
#endif

#ifdef MKL_GPU
void
bml_diagonalize_dense_gpu_double_real(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else

    int dnum = 0;
    MKL_INT info;
    const MKL_INT N = A->N;
    double *A_matrix = A->matrix;
    double *typed_eigenvalues = (double *) eigenvalues;

    double *evecs = (double *) calloc(A->N * A->N, sizeof(double));
    double *evals = (double *) calloc(A->N, sizeof(double));

#pragma omp target enter data map(alloc:evecs[0:N*N])
#pragma omp target enter data map(alloc:evals[0:N])

// pull from GPU
#pragma omp target update from(A_matrix[0:N*N])
    printf("Checking A matrix values \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %6.3f,", A_matrix[i * N + j]);
        }
        printf("\n");
    }
    // copy A to evecs on GPU
#pragma omp target teams distribute parallel for
    for (int i = 0; i < N * N; i++)
    {
        evecs[i] = A_matrix[i];
    }
/*
#pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++)
    {
        evals[i] = 1.0;
    }

#ifdef MKL_GPU_DEBUG

#pragma omp target update from(evecs[0:N*N])
#pragma omp target update from(evals[0:N])
    // Check values before syev
    printf("Checking evecs input values \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %6.3f,", evecs[i * N + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < N; i++)
    {
        printf("%d, %f \n", i, evals[i]);
    }
#endif // debug
*/
#ifdef BML_SYEVD
    // Divide and conquer solver
    const MKL_INT lwork = 1 + 6 * N + 2 * N * N;
    double *work = malloc(lwork * sizeof(double));
    const MKL_INT liwork = 3 + 5 * N;
    MKL_INT *iwork = malloc(liwork * sizeof(MKL_INT));
#pragma omp target enter data map(alloc:work[0:lwork])
#pragma omp target enter data map(alloc:iwork[0:liwork])

#pragma omp target variant dispatch device(dnum) use_device_ptr(evecs, evals, work, iwork)
    dsyevd("V", "U", &N, evecs, &N, evals, work, &lwork,
             iwork, &liwork, &info);
    free(iwork);
#else
    const MKL_INT lwork = 3 * N;
    double *work = calloc(lwork, sizeof(double));
#pragma omp target enter data map(alloc:work[0:lwork])
#pragma omp target variant dispatch device(dnum) use_device_ptr(evecs, evals, work)
    dsyev("V", "U", &N, evecs, &N, evals, work, &lwork, &info);
#endif // BML_SYEVD

#ifdef MKL_GPU
       // pull evecs, evals back from GPU
#pragma omp target update from(evecs[0:N*N])
#pragma omp target update from(evals[0:N])

    printf("After SYEV \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %6.3f,", evecs[i * N + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < N; i++)
    {
        printf("%d, %f \n", i, evals[i]);
    }
#endif

// need typed_eigenvalues on CPU
#pragma omp target update from(evals[0:N])
    for (int i = 0; i < A->N; i++)
    {
        typed_eigenvalues[i] = (double) evals[i];
    }

// leave eigenvectors on GPU
    double *e_matrix = (double *) eigenvectors->matrix;
#pragma omp target enter data map(alloc:e_matrix[0:N*N])
#pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            e_matrix[ROWMAJOR(i, j, N, N)] = evecs[COLMAJOR(i, j, N, N)];
        }
    }

// push eigenvectors back from GPU
// #pragma omp target update from(e_matrix[0:N*N])

    free(evecs);
    free(evals);
    free(work);
#endif // NOBLAS
}
#endif // MKL_GPU

void
bml_diagonalize_dense_cpu_double_real(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    int info;
    int N = A->N;
    double *A_matrix = A->matrix;
    double *typed_eigenvalues = (double *) eigenvalues;

    double *evecs = calloc(A->N * A->N, sizeof(double));
    double *evals = calloc(A->N, sizeof(double));

    memcpy(evecs, A->matrix, A->N * A->N * sizeof(double));

#ifdef BML_SYEVD
    // Divide and conquer solver
    int lwork = 1 + 6 * A->N + 2 * A->N * A->N;
    double *work = malloc(lwork * sizeof(double));
    int liwork = 3 + 5 * A->N;
    int *iwork = malloc(liwork * sizeof(int));
    C_DSYEVD("V", "U", &A->N, evecs, &A->N, evals, work, &lwork,
             iwork, &liwork, &info);
    free(iwork);
#else
    int lwork = 3 * A->N;
    double *work = calloc(lwork, sizeof(double));
    C_DSYEV("V", "U", &A->N, evecs, &A->N, evals, work, &lwork, &info);
#endif

    A_matrix = (double *) eigenvectors->matrix;
    for (int i = 0; i < A->N; i++)
    {
        typed_eigenvalues[i] = (double) evals[i];
        for (int j = 0; j < A->N; j++)
        {
            A_matrix[ROWMAJOR(i, j, A->N, A->N)] =
                evecs[COLMAJOR(i, j, A->N, A->N)];
        }
    }
    free(evecs);
    free(evals);
    free(work);
#endif // NOBLAS
}

void
bml_diagonalize_dense_double_real(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
#ifdef BML_USE_MAGMA
    bml_diagonalize_dense_magma_double_real(A, eigenvalues, eigenvectors);
#elif MKL_GPU
    bml_diagonalize_dense_gpu_double_real(A, eigenvalues, eigenvectors);
#else // CPU code
    bml_diagonalize_dense_cpu_double_real(A, eigenvalues, eigenvectors);
#endif
}

void
bml_diagonalize_dense_single_complex(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
    int info;
    int N = A->N;
    float complex *typed_eigenvalues = (float complex *) eigenvalues;

#ifdef BML_USE_MAGMA
    int nb = magma_get_ssytrd_nb(A->N);
    magmaFloatComplex *evecs;
    magma_int_t ret = magma_cmalloc(&evecs, A->N * A->ld);
    assert(ret == MAGMA_SUCCESS);

    float *evals = malloc(A->N * sizeof(float));
    int lwork = 2 * A->N + A->N * nb;
    int tmp = 1 + 6 * A->N + 2 * A->N * A->N;
    if (tmp > lwork)
        lwork = tmp;
    magmaFloatComplex *work = malloc(lwork * sizeof(magmaFloatComplex));
    int liwork = 3 + 5 * A->N;
    int *iwork = malloc(liwork * sizeof(int));
    int ldwa = A->ld;
    magmaFloatComplex *wa = malloc(A->N * ldwa * sizeof(magmaFloatComplex));
    int lrwork = 1 + 5 * A->N + 2 * A->N * A->N;
    float *rwork = malloc(lrwork * sizeof(float));

    //copy matrix into evecs
    magmablas_clacpy(MagmaFull, A->N, A->N, A->matrix, A->ld, evecs, A->ld,
                     bml_queue());

    magma_queue_sync(bml_queue());

    magma_cheevd_gpu(MagmaVec, MagmaUpper, A->N, evecs, A->ld, evals,
                     wa, ldwa,
                     work, lwork, rwork, lrwork, iwork, liwork, &info);
    if (info != 0)
        LOG_ERROR("ERROR in magma_cheevd_gpu");

    magmaFloatComplex *A_matrix = (magmaFloatComplex *) eigenvectors->matrix;
    magmablas_ctranspose(A->N, A->N, evecs, A->ld,
                         A_matrix, eigenvectors->ld, bml_queue());
    for (int i = 0; i < A->N; i++)
        typed_eigenvalues[i] = (double) evals[i];
    magma_queue_sync(bml_queue());

    magma_free(evecs);
    free(wa);
#else
    int *isuppz = calloc(2 * A->N, sizeof(int));
    int lwork = 2 * A->N;
    int liwork = 10 * A->N;
    int lrwork = 24 * A->N;
    int *iwork = calloc(liwork, sizeof(int));
    int M;
    float *evals = calloc(A->N, sizeof(float));
    float *rwork = calloc(lrwork, sizeof(float));
    float complex *A_copy = calloc(A->N * A->N, sizeof(float complex));
    float complex *A_matrix;
    float complex *evecs = calloc(A->N * A->N, sizeof(float complex));
    float complex *work = calloc(lwork, sizeof(float complex));

#ifdef MKL_GPU
// pull from GPU

#pragma omp target update from(A_matrix[0:N*N])
#endif
    memcpy(A_copy, A->matrix, A->N * A->N * sizeof(float complex));
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    {
        float abstol = 0;
        float vl = -FLT_MAX;
        float vu = FLT_MAX;
        int il = 1;
        int iu = A->N;
        C_CHEEVR("V", "A", "U", &A->N, A_copy, &A->N, &vl, &vu, &il, &iu,
                 &abstol, &M, evals, evecs, &A->N, isuppz, work, &lwork,
                 rwork, &lrwork, iwork, &liwork, &info);
    }
#endif
    A_matrix = (float complex *) eigenvectors->matrix;
    for (int i = 0; i < A->N; i++)
    {
        typed_eigenvalues[i] = (double) evals[i];
        for (int j = 0; j < A->N; j++)
        {
            A_matrix[ROWMAJOR(i, j, A->N, A->N)] =
                evecs[COLMAJOR(i, j, A->N, A->N)];
        }
    }

#ifdef MKL_GPU
// push back to GPU
#pragma omp target update to(A_matrix[0:N*N])
#endif
    free(A_copy);
    free(evecs);
    free(isuppz);
#endif
    free(work);
    free(rwork);
    free(iwork);
    free(evals);
}

void
bml_diagonalize_dense_double_complex(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_dense_t * eigenvectors)
{
    int info;
    int N = A->N;
    double complex *typed_eigenvalues = (double complex *) eigenvalues;

#ifdef BML_USE_MAGMA
    int nb = magma_get_ssytrd_nb(A->N);
    magmaDoubleComplex *evecs;
    magma_int_t ret = magma_zmalloc(&evecs, A->N * A->ld);
    assert(ret == MAGMA_SUCCESS);

    double *evals = malloc(A->N * sizeof(double));
    int lwork = 2 * A->N + A->N * nb;
    int tmp = 1 + 6 * A->N + 2 * A->N * A->N;
    if (tmp > lwork)
        lwork = tmp;
    magmaDoubleComplex *work = malloc(lwork * sizeof(magmaDoubleComplex));
    int liwork = 3 + 5 * A->N;
    int *iwork = malloc(liwork * sizeof(int));
    int ldwa = A->ld;
    magmaDoubleComplex *wa = malloc(A->N * ldwa * sizeof(magmaDoubleComplex));
    int lrwork = 1 + 5 * A->N + 2 * A->N * A->N;
    double *rwork = malloc(lrwork * sizeof(double));

    //copy matrix into evecs
    magmablas_zlacpy(MagmaFull, A->N, A->N, A->matrix, A->ld, evecs, A->ld,
                     bml_queue());
    magma_queue_sync(bml_queue());

    magma_zheevd_gpu(MagmaVec, MagmaUpper, A->N, evecs, A->ld, evals,
                     wa, ldwa,
                     work, lwork, rwork, lrwork, iwork, liwork, &info);
    if (info != 0)
        LOG_ERROR("ERROR in magma_csyevd_gpu");

    magmaDoubleComplex *A_matrix =
        (magmaDoubleComplex *) eigenvectors->matrix;
    magmablas_ztranspose(A->N, A->N, evecs, A->ld, A_matrix, eigenvectors->ld,
                         bml_queue());
    for (int i = 0; i < A->N; i++)
        typed_eigenvalues[i] = (double) evals[i];

    magma_queue_sync(bml_queue());

    magma_free(evecs);
    free(wa);
#else
    int *isuppz = calloc(2 * A->N, sizeof(int));
    int liwork = 10 * A->N;
    int lrwork = 24 * A->N;
    int lwork = 2 * A->N;
    int *iwork = calloc(liwork, sizeof(int));
    int M;
    double *evals = calloc(A->N, sizeof(double));
    double *rwork = calloc(lrwork, sizeof(double));
    double complex *A_copy = calloc(A->N * A->N, sizeof(double complex));
    double complex *A_matrix;
    double complex *evecs = calloc(A->N * A->N, sizeof(double complex));
    double complex *work = calloc(lwork, sizeof(double complex));

#ifdef MKL_GPU
// pull from GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif
    memcpy(A_copy, A->matrix, A->N * A->N * sizeof(double complex));
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    {
        double abstol = 0;
        double vl = -DBL_MAX;
        double vu = DBL_MAX;
        int il = 1;
        int iu = A->N;

        C_ZHEEVR("V", "A", "U", &A->N, A_copy, &A->N, &vl, &vu, &il, &iu,
                 &abstol, &M, evals, evecs, &A->N, isuppz, work, &lwork,
                 rwork, &lrwork, iwork, &liwork, &info);
    }
#endif
    A_matrix = (double complex *) eigenvectors->matrix;
    for (int i = 0; i < A->N; i++)
    {
        typed_eigenvalues[i] = (double) evals[i];
        for (int j = 0; j < A->N; j++)
        {
            A_matrix[ROWMAJOR(i, j, A->N, A->N)] =
                evecs[COLMAJOR(i, j, A->N, A->N)];
        }
    }

#ifdef MKL_GPU
// push back to GPU
#pragma omp target update to(A_matrix[0:N*N])
#endif
    free(A_copy);
    free(evecs);
    free(isuppz);
#endif
    free(work);
    free(rwork);
    free(iwork);
    free(evals);
}

void
bml_diagonalize_dense(
    bml_matrix_dense_t * A,
    void *eigenvalues,
    bml_matrix_t * eigenvectors)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_diagonalize_dense_single_real(A, eigenvalues, eigenvectors);
            break;
        case double_real:
            bml_diagonalize_dense_double_real(A, eigenvalues, eigenvectors);
            break;
#ifdef BML_COMPLEX
        case single_complex:
            bml_diagonalize_dense_single_complex(A, eigenvalues,
                                                 eigenvectors);
            break;
        case double_complex:
            bml_diagonalize_dense_double_complex(A, eigenvalues,
                                                 eigenvectors);
            break;
#endif
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}
