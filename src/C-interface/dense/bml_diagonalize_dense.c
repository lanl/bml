#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_types.h"
#include "../lapack.h"
#include "../macros.h"
#include "bml_allocate_dense.h"
#include "bml_diagonalize_dense.h"
#include "bml_types_dense.h"

#include <string.h>

/** \page diagonalize
 *
 * Note: We can't generify these functions easily since the API
 * differs between the real and complex types. rwork and lrwork are
 * only used in the complex cases. We opted instead to explicitly
 * implement the four versions.
 *
 * Note also that the API allows only for real eigenvectors. In the
 * complex cases, the eigenvectors are complex. */

void
bml_diagonalize_dense_single_real(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_dense_t ** eigenvectors)
{
    float *A_copy = calloc(A->N * A->N, sizeof(float));
    float *evecs = calloc(A->N * A->N, sizeof(float));
    int M;
    int *isuppz = calloc(2 * A->N, sizeof(int));
    int lwork = 2 * A->N;
    float *work = calloc(lwork, sizeof(float));
    int iwork = 10 * A->N;
    int *liwork = calloc(iwork, sizeof(int));
    int info;
    float abstol = 0;
    float *evals = calloc(A->N, sizeof(float));
    float *A_matrix;

    memcpy(A_copy, A->matrix, A->N * A->N * sizeof(float));
    C_SSYEVR("V", "A", "U", &A->N, A_copy, &A->N, NULL, NULL, NULL, NULL,
             &abstol, &M, evals, evecs, &A->N, isuppz, work, &lwork,
             &iwork, liwork, &info);

    *eigenvalues = bml_allocate_memory(A->N * sizeof(double));
    *eigenvectors = bml_zero_matrix_dense_single_real(A->N);
    A_matrix = (float *) (*eigenvectors)->matrix;
    for (int i = 0; i < A->N; i++)
    {
        (*eigenvalues)[i] = (double) evals[i];
        for (int j = 0; j < A->N; j++)
        {
            A_matrix[ROWMAJOR(i, j, A->N)] = evecs[ROWMAJOR(i, j, A->N)];
        }
    }

    free(A_copy);
    free(evecs);
    free(isuppz);
    free(work);
    free(liwork);
    free(evals);
}

void
bml_diagonalize_dense_double_real(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_dense_t ** eigenvectors)
{
    double *A_copy = calloc(A->N * A->N, sizeof(double));
    double *evecs = calloc(A->N * A->N, sizeof(double));
    int M;
    int *isuppz = calloc(2 * A->N, sizeof(int));
    int lwork = 2 * A->N;
    double *work = calloc(lwork, sizeof(double));
    int iwork = 10 * A->N;
    int *liwork = calloc(iwork, sizeof(int));
    int info;
    double abstol = 0;
    double *evals = calloc(A->N, sizeof(double));
    double *A_matrix;

    memcpy(A_copy, A->matrix, A->N * A->N * sizeof(double));
    C_DSYEVR("V", "A", "U", &A->N, A_copy, &A->N, NULL, NULL, NULL, NULL,
             &abstol, &M, evals, evecs, &A->N, isuppz, work, &lwork,
             &iwork, liwork, &info);

    *eigenvalues = bml_allocate_memory(A->N * sizeof(double));
    *eigenvectors = bml_zero_matrix_dense_double_real(A->N);
    A_matrix = (double *) (*eigenvectors)->matrix;
    for (int i = 0; i < A->N; i++)
    {
        (*eigenvalues)[i] = (double) evals[i];
        for (int j = 0; j < A->N; j++)
        {
            A_matrix[ROWMAJOR(i, j, A->N)] = evecs[ROWMAJOR(i, j, A->N)];
        }
    }

    free(A_copy);
    free(evecs);
    free(isuppz);
    free(work);
    free(liwork);
    free(evals);
}

void
bml_diagonalize_dense_single_complex(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_dense_t ** eigenvectors)
{
    float complex *A_copy = calloc(A->N * A->N, sizeof(float complex));
    float complex *evecs = calloc(A->N * A->N, sizeof(float complex));
    int M;
    int *isuppz = calloc(2 * A->N, sizeof(int));
    int lwork = 2 * A->N;
    float complex *work = calloc(lwork, sizeof(float complex));
    int lrwork = 24 * A->N;
    int iwork = 10 * A->N;
    int *liwork = calloc(iwork, sizeof(int));
    int info;
    float abstol = 0;
    float *evals = calloc(A->N, sizeof(float));
    float *rwork = calloc(lrwork, sizeof(float));
    float complex *A_matrix;

    memcpy(A_copy, A->matrix, A->N * A->N * sizeof(float complex));
    C_CHEEVR("V", "A", "U", &A->N, A_copy, &A->N, NULL, NULL, NULL, NULL,
             &abstol, &M, evals, evecs, &A->N, isuppz, work, &lwork, rwork,
             &lrwork, &iwork, liwork, &info);

    *eigenvalues = bml_allocate_memory(A->N * sizeof(double));
    *eigenvectors = bml_zero_matrix_dense_single_complex(A->N);
    A_matrix = (float complex *) (*eigenvectors)->matrix;
    for (int i = 0; i < A->N; i++)
    {
        (*eigenvalues)[i] = (double) evals[i];
        for (int j = 0; j < A->N; j++)
        {
            A_matrix[ROWMAJOR(i, j, A->N)] = evecs[ROWMAJOR(i, j, A->N)];
        }
    }

    free(A_copy);
    free(evecs);
    free(isuppz);
    free(work);
    free(rwork);
    free(liwork);
    free(evals);
}

void
bml_diagonalize_dense_double_complex(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_dense_t ** eigenvectors)
{
    double complex *A_copy = calloc(A->N * A->N, sizeof(double complex));
    double complex *evecs = calloc(A->N * A->N, sizeof(double complex));
    int M;
    int *isuppz = calloc(2 * A->N, sizeof(int));
    int lwork = 2 * A->N;
    double complex *work = calloc(lwork, sizeof(double complex));
    int lrwork = 24 * A->N;
    int iwork = 10 * A->N;
    int *liwork = calloc(iwork, sizeof(int));
    int info;
    double abstol = 0;
    double *evals = calloc(A->N, sizeof(double));
    double *rwork = calloc(lrwork, sizeof(double));
    double complex *A_matrix;

    memcpy(A_copy, A->matrix, A->N * A->N * sizeof(double complex));
    C_ZHEEVR("V", "A", "U", &A->N, A_copy, &A->N, NULL, NULL, NULL, NULL,
             &abstol, &M, evals, evecs, &A->N, isuppz, work, &lwork, rwork,
             &lrwork, &iwork, liwork, &info);

    *eigenvalues = bml_allocate_memory(A->N * sizeof(double));
    *eigenvectors = bml_zero_matrix_dense_double_complex(A->N);
    A_matrix = (double complex *) (*eigenvectors)->matrix;
    for (int i = 0; i < A->N; i++)
    {
        (*eigenvalues)[i] = (double) evals[i];
        for (int j = 0; j < A->N; j++)
        {
            A_matrix[ROWMAJOR(i, j, A->N)] = evecs[ROWMAJOR(i, j, A->N)];
        }
    }

    free(A_copy);
    free(evecs);
    free(isuppz);
    free(work);
    free(rwork);
    free(liwork);
    free(evals);
}

void
bml_diagonalize_dense(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_t ** eigenvectors)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_diagonalize_dense_single_real(A, eigenvalues, eigenvectors);
            break;
        case double_real:
            bml_diagonalize_dense_double_real(A, eigenvalues, eigenvectors);
            break;
        case single_complex:
            bml_diagonalize_dense_single_complex(A, eigenvalues,
                                                 eigenvectors);
            break;
        case double_complex:
            bml_diagonalize_dense_double_complex(A, eigenvalues,
                                                 eigenvectors);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }

}
