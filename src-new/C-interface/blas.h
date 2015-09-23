/** \file */

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

#endif
