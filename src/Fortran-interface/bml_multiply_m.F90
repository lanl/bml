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

  end interface

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

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a, b
    type(bml_matrix_t), intent(inout) :: c
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta
    double precision, optional, intent(in) :: threshold 

    call bml_multiply_c(a%ptr, b%ptr, c%ptr, alpha, beta, threshold)

  end subroutine bml_multiply

end module bml_multiply_m
