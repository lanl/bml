!> Matrix multiplication.
module bml_multiply_m

  implicit none

  private

  interface

     subroutine bml_multiply_C(a, b, c, alpha, beta, threshold) &
          bind(C, name="bml_multiply")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       type(C_PTR), value, intent(in) :: b
       type(C_PTR), value, intent(in) :: c
       real(C_DOUBLE), value, intent(in) :: alpha
       real(C_DOUBLE), value, intent(in) :: beta
       real(C_DOUBLE), value, intent(in) :: threshold
     end subroutine bml_multiply_C

     subroutine bml_multiply_x2_C(a, b, threshold) &
          bind(C, name="bml_multiply_x2")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       type(C_PTR), value, intent(in) :: b
       real(C_DOUBLE), value, intent(in) :: threshold
     end subroutine bml_multiply_x2_C

  end interface

  interface bml_multiply
     module procedure bml_multiply
  end interface

  interface bml_multiply_x2
     module procedure bml_multiply_x2
  end interface

  public :: bml_multiply
  public :: bml_multiply_x2

contains

  !> Multiply two matrices.
  !!
  !! \ingroup multiply_group
  !!
  !! \f$ C \leftarrow \alpha A \times B + \beta C \f$
  !!
  !! The optional scaling factors \f$ \alpha \f$ and \f$ \beta \f$
  !! default to \f$ \alpha = 1 \f$ and \f$ \beta = 0 \f$.
  !!
  !! \param a Matrix \f$ A \f$.
  !! \param b Matrix \f$ B \f$.
  !! \param c Matrix \f$ C \f$.
  !! \param alpha The factor \f$ \alpha \f$.
  !! \param beta The factor \f$ \beta \f$.
  !! \param threshold The threshold \f$ threshold \f$.
  subroutine bml_multiply(a, b, c, alpha, beta, threshold)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a, b
    type(bml_matrix_t), intent(inout) :: c
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta
    double precision, optional, intent(in) :: threshold

    double precision :: alpha_
    double precision :: beta_
    double precision :: threshold_

    if(present(alpha)) then
       alpha_ = alpha
    else
       alpha_ = 1
    end if
    if(present(beta)) then
       beta_ = beta
    else
       beta_ = 0
    end if
    if(present(threshold)) then
       threshold_ = threshold
    else
       threshold_ = 0
    end if
    call bml_multiply_c(a%ptr, b%ptr, c%ptr, alpha_, beta_, threshold_)

  end subroutine bml_multiply

  !> Square a matrix.
  !!
  !! \ingroup multiply_group
  !!
  !! \f$B \leftarrow A \times A \f$
  !!
  !! \param a Matrix \f$ A \f$.
  !! \param b Matrix \f$ B \f$.
  !! \param threshold The threshold \f$ threshold \f$.
  subroutine bml_multiply_x2(a, b, threshold)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: b
    double precision, optional, intent(in) :: threshold

    double precision :: threshold_

    if(present(threshold)) then
       threshold_ = threshold
    else
       threshold_ = 0
    end if

    call bml_multiply_x2_c(a%ptr, b%ptr, threshold_)

  end subroutine bml_multiply_x2

end module bml_multiply_m
