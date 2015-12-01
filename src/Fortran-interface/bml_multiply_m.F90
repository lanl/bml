!> Matrix multiplication.
module bml_multiply_m
  use bml_c_interface_m
  use bml_types_m
  implicit none

  private

  public :: bml_multiply

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

    type(bml_matrix_t), intent(in) :: a, b
    type(bml_matrix_t), intent(inout) :: c
    real(C_DOUBLE), optional, intent(in) :: alpha
    real(C_DOUBLE), optional, intent(in) :: beta
    real(C_DOUBLE), optional, intent(in) :: threshold

    real(C_DOUBLE) :: alpha_
    real(C_DOUBLE) :: beta_
    real(C_DOUBLE) :: threshold_

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

end module bml_multiply_m
