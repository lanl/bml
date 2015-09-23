#ifndef __TYPED_H
#define __TYPED_H

#if defined(SINGLE_REAL)
#define REAL_T float
#define FUNC_SUFFIX single_real
#define MATRIX_PRECISION single_real
#elif defined(DOUBLE_REAL)
#define REAL_T double
#define FUNC_SUFFIX double_real
#define MATRIX_PRECISION double_real
#else
#error Unknown precision type
#endif

#define CONCAT2(a, b) a ## _ ## b
#define CONCAT(a, b) CONCAT2(a, b)
#define TYPED_FUNC(a) CONCAT(a, FUNC_SUFFIX)

#endif
