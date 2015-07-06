!> \copyright Los Alamos National Laboratory 2015

!> Matrix addition for dense matrices.
module bml_add_dense
  implicit none
contains

  !> Add two dense matrices.
  !!
  !! \f$ C \leftarrow \alpha A + \beta B \f$
  !!
  !! \param A Matrix \f$ A \f$.
  !! \param B Matrix \f$ B \f$.
  !! \param C Matrix \f$ C \f$.
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_dense(A, B, C, alpha, beta)

    use bml_type_dense

    type(bml_matrix_dense_t), intent(in) :: A, B
    type(bml_matrix_dense_t), intent(inout) :: C
    double precision, optional :: alpha, beta
    double precision :: alpha_, beta_

    if(present(alpha)) then
       alpha_ = alpha
    else
       alpha_ = 1
    endif

    if(present(beta)) then
       beta_ = beta
    else
       beta_ = 1
    endif

    C%matrix = alpha_*A%matrix+beta_*B%matrix

  end subroutine add_dense

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! \param A Matrix A
  !! \param C Matrix C
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_dense(A, C, alpha, beta)

    use bml_type_dense

    type(bml_matrix_dense_t), intent(in) :: A
    type(bml_matrix_dense_t), intent(out) :: C
    double precision, optional, intent(in) :: alpha
    double precision, optional, intent(in) :: beta

  end subroutine add_identity_dense

end module bml_add_dense
