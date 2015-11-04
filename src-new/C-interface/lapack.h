#ifndef __LAPACK_H
#    define __LAPACK_H

void C_SSYEV(
    const char *jobz,
    const char *uplo,
    const int *n,
    float *a,
    const int *lda,
    float *w,
    float *work,
    const int *lwork,
    int *info);
void C_DSYEV(
    const char *jobz,
    const char *uplo,
    const int *n,
    double *a,
    const int *lda,
    double *w,
    double *work,
    const int *lwork,
    int *info);

#endif
