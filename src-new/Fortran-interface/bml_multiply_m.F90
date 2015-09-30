!> Matrix multiplication.
module bml_multiply_m
  implicit none
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
  subroutine bml_multiply(a, b, c, alpha, beta)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a, b
    type(bml_matrix_t), intent(inout) :: c
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta

  end subroutine bml_multiply

end module bml_multiply_m
