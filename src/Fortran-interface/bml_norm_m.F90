!> Norm procedures.
module bml_norm_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_sum_squares
  public :: bml_sum_squares2
  public :: bml_fnorm

contains

  !> Return sum of squares of the elements of a matrix.
  !! 
  !!\param a The matrix
  !!\return The sum of squares
  function bml_sum_squares(a) result(sum)

    type(bml_matrix_t), intent(in) :: a
    real(C_DOUBLE) :: sum

    sum = bml_sum_squares_C(a%ptr)

  end function bml_sum_squares

  !> Return sum of squares of alpha * A + beta * B.
  !! 
  !!\param a The matrix a
  !!\param b The matrix b
  !!\param alpha Multiplier for a
  !!\param beta Multiplier for b
  !!\return The sum of squares for alpha * A + beta * B
  function bml_sum_squares2(a, b, alpha, beta) result(sum2)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(in) :: b
    real(C_DOUBLE), intent(in) :: alpha
    real(C_DOUBLE), intent(in) :: beta
    real(C_DOUBLE) :: sum2

    sum2 = bml_sum_squares2_C(a%ptr, b%ptr, alpha, beta)

  end function bml_sum_squares2

  !> Return Frobenius norm of a matrix.
  !! 
  !!\param a The matrix
  !!\return The Frobenius norm
  function bml_fnorm(a) result(fnorm)

    type(bml_matrix_t), intent(in) :: a
    real(C_DOUBLE) :: fnorm

    fnorm = bml_fnorm_C(a%ptr)

  end function bml_fnorm

end module bml_norm_m
