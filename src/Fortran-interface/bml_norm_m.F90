!> Norm procedures.
module bml_norm_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_sum_squares
  public :: bml_sum_squares_submatrix
  public :: bml_sum_squares2
  public :: bml_fnorm

contains

  !> Return sum of squares of the elements of a matrix.
  !!
  !!\param a The matrix
  !!\return The sum of squares
  function bml_sum_squares(a) result(ssum)

    type(bml_matrix_t), intent(in) :: a
    real(C_DOUBLE) :: ssum

    ssum = bml_sum_squares_C(a%ptr)

  end function bml_sum_squares

  !> Return sum of squares of the core elements of a submatrix.
  !!
  !!\param a The matrix
  !!\param core_pos Core rows of submatrix
  !!\param core_size Number of rows
  !!\return The sum of squares
  function bml_sum_squares_submatrix(a, core_pos, core_size) result(ssum)

    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), target, intent(in) :: core_pos(*)
    integer(C_INT), intent(in) :: core_size
    real(C_DOUBLE) :: ssum

    ssum = bml_sum_squares_submatrix_C(a%ptr, c_loc(core_pos), core_size)

  end function bml_sum_squares_submatrix

  !> Return sum of squares of alpha * A + beta * B.
  !!
  !!\param a The matrix a
  !!\param b The matrix b
  !!\param alpha Multiplier for a
  !!\param beta Multiplier for b
  !!\return The sum of squares for alpha * A + beta * B
  function bml_sum_squares2(a, b, alpha, beta, threshold) result(ssum2)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(in) :: b
    real(C_DOUBLE), intent(in) :: alpha
    real(C_DOUBLE), intent(in) :: beta
    real(C_DOUBLE), intent(in) :: threshold
    real(C_DOUBLE) :: ssum2

    ssum2 = bml_sum_squares2_C(a%ptr, b%ptr, alpha, beta, threshold)

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
