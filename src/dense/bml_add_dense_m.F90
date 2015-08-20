!> \copyright Los Alamos National Laboratory 2015

!> Matrix addition for dense matrices.
module bml_add_dense_m
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

    use bml_type_dense_m

    type(bml_matrix_dense_t), intent(in) :: A, B
    type(bml_matrix_dense_t), intent(inout) :: C
    double precision, intent(in) :: alpha, beta

    C%matrix = alpha*A%matrix+beta*B%matrix

  end subroutine add_dense

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! \param A Matrix A
  !! \param C Matrix C
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_two_dense(A, C, alpha, beta)

    use bml_type_dense_m

    type(bml_matrix_dense_t), intent(in) :: A
    type(bml_matrix_dense_t), intent(out) :: C
    double precision, intent(in) :: alpha
    double precision, intent(in) :: beta

    integer :: i

    C%matrix = alpha*A%matrix
    do i = 1, A%N
       C%matrix(i, i) = C%matrix(i, i) + beta
    end do

  end subroutine add_identity_two_dense

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! \param A Matrix A
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_self_dense(A, alpha, beta)

    use bml_type_dense_m

    type(bml_matrix_dense_t), intent(inout) :: A
    double precision, intent(in) :: alpha
    double precision, intent(in) :: beta

    integer :: i

    A%matrix = alpha*A%matrix
    do i = 1, A%N
       A%matrix(i, i) = A%matrix(i, i) + beta
    end do

  end subroutine add_identity_self_dense

end module bml_add_dense_m
