#ifndef __TYPED_H
#define __TYPED_H

#if defined(SINGLE_REAL)
#define REAL_T float
#define FUNC_SUFFIX single_real
#define MATRIX_PRECISION single_real
#define BLAS_PREFIX S
#define REAL_PART(x) (x)
#define is_above_threshold(x, t) (fabsf(x) > (float) (t))
#elif defined(DOUBLE_REAL)
#define REAL_T double
#define FUNC_SUFFIX double_real
#define MATRIX_PRECISION double_real
#define BLAS_PREFIX D
#define REAL_PART(x) (x)
#define is_above_threshold(x, t) (fabs(x) > (t))
#elif defined(SINGLE_COMPLEX)
#define REAL_T float complex
#define FUNC_SUFFIX single_complex
#define MATRIX_PRECISION single_complex
#define BLAS_PREFIX C
#define REAL_PART(x) crealf(x)
#define is_above_threshold(x, t) (cabsf(x) > cabsf((float) (t)))
#elif defined(DOUBLE_COMPLEX)
#define REAL_T double complex
#define FUNC_SUFFIX double_complex
#define MATRIX_PRECISION double_complex
#define BLAS_PREFIX Z
#define REAL_PART(x) creal(x)
#define is_above_threshold(x, t) (cabs(x) > cabs(t))
#else
#error Unknown precision type
#endif

#define CONCAT2_(a, b) a ## _ ## b
#define CONCAT_(a, b) CONCAT2_(a, b)

#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)

#define TYPED_FUNC(a) CONCAT_(a, FUNC_SUFFIX)
#define C_BLAS(a) CONCAT_(C, CONCAT(BLAS_PREFIX , a))

#endif
