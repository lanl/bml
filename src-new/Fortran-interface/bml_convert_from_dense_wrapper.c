#include "bml_convert.h"

/* function bml_convert_from_dense_single_C(matrix_type, matrix_precision, n, a, threshold) & */
/*      bind(C, name="bml_convert_from_dense_wrapper_single") */
/*   use, intrinsic :: iso_C_binding */
/*   integer(C_INT), value, intent(in) :: matrix_type */
/*   integer(C_INT), value, intent(in) :: matrix_precision */
/*   integer(C_INT), value, intent(in) :: n */
/*   real(C_FLOAT), intent(in) :: a(:, :) */
/*   real(C_DOUBLE), value, intent(in) :: threshold */
/*   type(C_PTR) :: bml_convert_from_dense_single_C */
/* end function bml_convert_from_dense_single_C */

void *bml_convert_from_dense_wrapper_single(int matrix_type,
                                            int matrix_precision,
                                            int N,
                                            float *A,
                                            double threshold)
{
    return bml_convert_from_dense(matrix_type, matrix_precision, N, A, threshold);
}

/* function bml_convert_from_dense_double_C(matrix_type, matrix_precision, n, a, threshold) & */
/*      bind(C, name="bml_convert_from_dense_wrapper_double") */
/*   use, intrinsic :: iso_C_binding */
/*   integer(C_INT), value, intent(in) :: matrix_type */
/*   integer(C_INT), value, intent(in) :: matrix_precision */
/*   integer(C_INT), value, intent(in) :: n */
/*   real(C_DOUBLE), intent(in) :: a(:, :) */
/*   real(C_DOUBLE), value, intent(in) :: threshold */
/*   type(C_PTR) :: bml_convert_from_dense_double_C */
/* end function bml_convert_from_dense_double_C */

void *bml_convert_from_dense_wrapper_double(int matrix_type,
                                            int matrix_precision,
                                            int N,
                                            double *A,
                                            double threshold)
{
    return bml_convert_from_dense(matrix_type, matrix_precision, N, A, threshold);
}
