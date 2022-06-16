#ifndef __BML_TYPED_H
#define __BML_TYPED_H

#include <float.h>

/* Fix the generated function name. */
#if defined(SINGLE_REAL)
#define FUNC_SUFFIX single_real
#elif defined(DOUBLE_REAL)
#define FUNC_SUFFIX double_real
#elif defined(SINGLE_COMPLEX)
#define FUNC_SUFFIX single_complex
#elif defined(DOUBLE_COMPLEX)
#define FUNC_SUFFIX double_complex
#else
#error Unknown precision type
#endif

/* Define numeric types. */
#if defined(SINGLE_REAL)
#define REAL_T float
#define MAGMA_T float
#define MKL_T float
#define MKL_REAL(a) a
#define MKL_IMAG(a) a
#define MKL_ADDRESS(a) a
#define MPI_T MPI_FLOAT
#define MATRIX_PRECISION single_real
#define BLAS_PREFIX S
#define MAGMA_PREFIX s
#define XSMM_PREFIX libxsmm_
#define REAL_PART(x) (x)
#define IMAGINARY_PART(x) (0.0)
#define COMPLEX_CONJUGATE(x) (x)
#define ABS(x) (fabsf(x))
#define is_above_threshold(x, t) (fabsf(x) > (float) (t))
/* floating point limit types */
#define BML_REAL_MAX FLT_MAX
#define BML_REAL_MIN FLT_MIN
#define BML_REAL_EPSILON FLT_EPSILON
/* cusparse types */
#if defined (BML_USE_CUSPARSE)
#define BML_CUSPARSE_T CUDA_R_32F
#define bml_cusparseCSRgeam2_bufferSizeExt cusparseScsrgeam2_bufferSizeExt
#define bml_cusparseCSRgeam2 cusparseScsrgeam2
#define bml_cusparsePruneCSR_bufferSizeExt cusparseSpruneCsr2csr_bufferSizeExt
#define bml_cusparsePruneCSRNnz cusparseSpruneCsr2csrNnz
#define bml_cusparsePruneCSR cusparseSpruneCsr2csr
#elif defined (BML_USE_ROCSPARSE)
#define BML_ROCSPARSE_T rocsparse_datatype_f32_r
#define bml_rocsparse_csrgeam_buffer_size rocsparse_scsrgeam_buffer_size
#define bml_rocsparse_csrgeam rocsparse_scsrgeam
#define bml_rocsparse_xprune_csr2csr_buffer_size rocsparse_sprune_csr2csr_buffer_size
#define bml_rocsparse_xprune_csr2csr_nnz rocsparse_sprune_csr2csr_nnz
#define bml_rocsparse_xprune_csr2csr rocsparse_sprune_csr2csr
#endif
#elif defined(DOUBLE_REAL)
#define REAL_T double
#define MAGMA_T double
#define MKL_T  double
#define MKL_REAL(a) a
#define MKL_IMAG(a) a
#define MKL_ADDRESS(a) a
#define MPI_T MPI_DOUBLE
#define MATRIX_PRECISION double_real
#define BLAS_PREFIX D
#define MAGMA_PREFIX d
#define XSMM_PREFIX libxsmm_
#define REAL_PART(x) (x)
#define IMAGINARY_PART(x) (0.0)
#define COMPLEX_CONJUGATE(x) (x)
#define ABS(x) (fabs(x))
#define is_above_threshold(x, t) (fabs(x) > (double) (t))
/* floating point limit types */
#define BML_REAL_MAX DBL_MAX
#define BML_REAL_MIN DBL_MIN
#define BML_REAL_EPSILON DBL_EPSILON
/* cusparse types */
#if defined (BML_USE_CUSPARSE)
#define BML_CUSPARSE_T CUDA_R_64F
#define bml_cusparseCSRgeam2_bufferSizeExt cusparseDcsrgeam2_bufferSizeExt
#define bml_cusparseCSRgeam2 cusparseDcsrgeam2
#define bml_cusparsePruneCSR_bufferSizeExt cusparseDpruneCsr2csr_bufferSizeExt
#define bml_cusparsePruneCSRNnz cusparseDpruneCsr2csrNnz
#define bml_cusparsePruneCSR cusparseDpruneCsr2csr
#elif defined (BML_USE_ROCSPARSE)
#define BML_ROCSPARSE_T rocsparse_datatype_f64_r
#define bml_rocsparse_csrgeam_buffer_size rocsparse_dcsrgeam_buffer_size
#define bml_rocsparse_csrgeam rocsparse_dcsrgeam
#define bml_rocsparse_xprune_csr2csr_buffer_size rocsparse_dprune_csr2csr_buffer_size
#define bml_rocsparse_xprune_csr2csr_nnz rocsparse_dprune_csr2csr_nnz
#define bml_rocsparse_xprune_csr2csr rocsparse_dprune_csr2csr
#endif
#elif defined(SINGLE_COMPLEX)
#define REAL_T float _Complex
#define MAGMA_T magmaFloatComplex
#define MKL_T MKL_Complex8
#define MKL_REAL(a) a.real
#define MKL_IMAG(a) a.imag=0.0
#define MKL_ADDRESS(a) &a
#define MPI_T MPI_C_FLOAT_COMPLEX
#define MATRIX_PRECISION single_complex
#define BLAS_PREFIX C
#define MAGMA_PREFIX c
#define XSMM_PREFIX
#define REAL_PART(x) (crealf(x))
#define IMAGINARY_PART(x) (cimagf(x))
#define COMPLEX_CONJUGATE(x) (conjf(x))
#define ABS(x) (cabsf(x))
#define is_above_threshold(x, t) (cabsf(x) > (float) (t))
/* floating point limit types */
#define BML_REAL_MAX FLT_MAX
#define BML_REAL_MIN FLT_MIN
#define BML_REAL_EPSILON FLT_EPSILON
/* cusparse types */
/* Note: as of 6/16/22 there is no complex prune function, these are placeholders */
#if defined (BML_USE_CUSPARSE)
#define BML_CUSPARSE_T CUDA_C_32F
#define bml_cusparseCSRgeam2_bufferSizeExt cusparseCcsrgeam2_bufferSizeExt
#define bml_cusparseCSRgeam2 cusparseCcsrgeam2
#define bml_cusparsePruneCSR_bufferSizeExt cusparseCpruneCsr2csr_bufferSizeExt
#define bml_cusparsePruneCSRNnz cusparseCpruneCsr2csrNnz
#define bml_cusparsePruneCSR cusparseCpruneCsr2csr
#elif defined (BML_USE_ROCSPARSE)
#define BML_ROCSPARSE_T rocsparse_datatype_f32_c
#define bml_rocsparse_csrgeam_buffer_size rocsparse_ccsrgeam_buffer_size
#define bml_rocsparse_csrgeam rocsparse_ccsrgeam
#define bml_rocsparse_xprune_csr2csr_buffer_size rocsparse_cprune_csr2csr_buffer_size
#define bml_rocsparse_xprune_csr2csr_nnz rocsparse_cprune_csr2csr_nnz
#define bml_rocsparse_xprune_csr2csr rocsparse_cprune_csr2csr
#endif
#elif defined(DOUBLE_COMPLEX)
#define REAL_T double _Complex
#define MAGMA_T magmaDoubleComplex
#define MKL_T MKL_Complex16
#define MKL_REAL(a) a.real
#define MKL_IMAG(a) a.imag=0.0
#define MKL_ADDRESS(a) &a

#define MPI_T MPI_C_DOUBLE_COMPLEX
#define MATRIX_PRECISION double_complex
#define BLAS_PREFIX Z
#define MAGMA_PREFIX z
#define XSMM_PREFIX
#define REAL_PART(x) (creal(x))
#define IMAGINARY_PART(x) (cimag(x))
#define COMPLEX_CONJUGATE(x) (conj(x))
#define ABS(x) (cabs(x))
#define is_above_threshold(x, t) (cabs(x) > (double) (t))
/* floating point limit types */
#define BML_REAL_MAX DBL_MAX
#define BML_REAL_MIN DBL_MIN
#define BML_REAL_EPSILON DBL_EPSILON
/* cusparse types */
#if defined (BML_USE_CUSPARSE)
#define BML_CUSPARSE_T CUDA_C_64F
#define bml_cusparseCSRgeam2_bufferSizeExt cusparseZcsrgeam2_bufferSizeExt
#define bml_cusparseCSRgeam2 cusparseZcsrgeam2
#define bml_cusparsePruneCSR_bufferSizeExt cusparseZpruneCsr2csr_bufferSizeExt
#define bml_cusparsePruneCSRNnz cusparseZpruneCsr2csrNnz
#define bml_cusparsePruneCSR cusparseZpruneCsr2csr
#elif defined (BML_USE_ROCSPARSE)
#define BML_ROCSPARSE_T rocsparse_datatype_f64_c
#define bml_rocsparse_csrgeam_buffer_size rocsparse_zcsrgeam_buffer_size
#define bml_rocsparse_csrgeam rocsparse_zcsrgeam
#define bml_rocsparse_xprune_csr2csr_buffer_size rocsparse_zprune_csr2csr_buffer_size
#define bml_rocsparse_xprune_csr2csr_nnz rocsparse_zprune_csr2csr_nnz
#define bml_rocsparse_xprune_csr2csr rocsparse_zprune_csr2csr
#endif
#else
#error Unknown precision type
#endif

#define CONCAT2_(a, b) a ## _ ## b
#define CONCAT_(a, b) CONCAT2_(a, b)

#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)

#define TYPED_FUNC(a) CONCAT_(a, FUNC_SUFFIX)
#define G_LAPACK(a) CONCAT(MAGMA_PREFIX , a)
//#define G_LAPACK(a) CONCAT_(LAPACKE, CONCAT(MAGMA_PREFIX , a))
#define G_BLAS(a) CONCAT_(cblas, CONCAT(MAGMA_PREFIX , a))
#define C_BLAS(a) CONCAT_(C, CONCAT(BLAS_PREFIX , a))
#define XSMM(a) CONCAT(XSMM_PREFIX , a)
#define MAGMACOMPLEX(a) CONCAT_(MAGMA, CONCAT_(BLAS_PREFIX, a))
#define MAGMA(a) CONCAT_(magma, CONCAT(MAGMA_PREFIX , a))
#define MAGMAGPU(a) CONCAT_(magma, CONCAT(MAGMA_PREFIX , CONCAT_(a, gpu)))
#define MAGMABLAS(a) CONCAT_(magmablas, CONCAT(MAGMA_PREFIX , a))

#if defined(BML_USE_CUSPARSE)
/* includes needed for use of printf (end EXIT_FAILURE if used.
 * May be used in void functions, hence return void.
 */
#include "stdlib.h"
#include "stdio.h"
#define BML_CHECK_CUSPARSE(func)                                               \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}
#elif defined(BML_USE_ROCSPARSE)
/* includes needed for use of printf (end EXIT_FAILURE if used.
 * May be used in void functions, hence return void.
 */
#include "stdlib.h"
#include "stdio.h"
#define BML_CHECK_ROCSPARSE(func)                                               \
{                                                                              \
    rocsparse_status status = (func);                                          \
    if (status != rocsparse_status_success) {                                   \
        printf("ROCPARSE API failed at line %d with error: %d\n",         \
               __LINE__, status);              \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}
#endif
#endif
