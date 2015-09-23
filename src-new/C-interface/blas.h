#ifndef __BLAS_H
#define __BLAS_H

void C_SSCAL(
    const int *n,
    const float *a,
    float *x,
    const int *incx);
void C_DSCAL(
    const int *n,
    const double *a,
    double *x,
    const int *incx);
void C_SGEMM(
    const char *transa,
    const char *transb,
    const int *m,
    const int *n,
    const int *k,
    const float *alpha,
    const float *a,
    const int *lda,
    const float *b,
    const int *ldb,
    const float *beta,
    float *c,
    const int *ldc);
void C_DGEMM(
    const char *transa,
    const char *transb,
    const int *m,
    const int *n,
    const int *k,
    const double *alpha,
    const double *a,
    const int *lda,
    const double *b,
    const int *ldb,
    const double *beta,
    double *c,
    const int *ldc);
void C_SAXPY(
    const int *n,
    const float *alpha,
    const float *x,
    const int *incx,
    float *y,
    const int *incy);
void C_DAXPY(
    const int *n,
    const double *alpha,
    const double *x,
    const int *incx,
    double *y,
    const int *incy);

#endif
