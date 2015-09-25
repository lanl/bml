#ifndef __TYPED_H
#define __TYPED_H

#if defined(SINGLE_REAL)
#define REAL_T float
#define FUNC_SUFFIX single_real
#define MATRIX_PRECISION single_real
#define BLAS_PREFIX S
#elif defined(DOUBLE_REAL)
#define REAL_T double
#define FUNC_SUFFIX double_real
#define MATRIX_PRECISION double_real
#define BLAS_PREFIX D
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
