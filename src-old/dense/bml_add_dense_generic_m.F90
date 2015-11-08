!> \copyright Los Alamos National Laboratory 2015

!> Matrix addition for dense matrices.
module bml_add_dense_BML_PRECISION_NAME_m

  implicit none

  interface add_two_dense
     module procedure add_two_dense_BML_PRECISION_NAME
  end interface add_two_dense

  interface add_three_dense
     module procedure add_three_dense_BML_PRECISION_NAME
  end interface add_three_dense

  interface add_identity_two_dense
     module procedure add_identity_two_dense_BML_PRECISION_NAME
  end interface add_identity_two_dense

  interface add_identity_self_dense
     module procedure add_identity_self_dense_BML_PRECISION_NAME
  end interface add_identity_self_dense

contains

  !> Add two dense matrices.
  !!
  !! \f$ C \leftarrow \alpha A + \beta B \f$
  !!
  !! \param a Matrix \f$ A \f$.
  !! \param b Matrix \f$ B \f$.
  !! \param c Matrix \f$ C \f$.
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_three_dense_BML_PRECISION_NAME(a, b, c, alpha, beta)

    use bml_type_dense_m

    type(bml_matrix_dense_BML_PRECISION_NAME_t), intent(in) :: a, b
    type(bml_matrix_dense_BML_PRECISION_NAME_t), intent(inout) :: c
    BML_REAL, intent(in) :: alpha, beta

    c%matrix = alpha*a%matrix+beta*b%matrix

  end subroutine add_three_dense_BML_PRECISION_NAME

  !> Add two dense matrices.
  !!
  !! \f$ A \leftarrow \alpha A + \beta B \f$
  !!
  !! \param alpha Factor \f$ \alpha \f$
  !! \param a Matrix \f$ A \f$.
  !! \param beta Factor \f$ \beta \f$
  !! \param b Matrix \f$ B \f$.
  subroutine add_two_dense_BML_PRECISION_NAME(alpha, a, beta, b)

    use bml_type_dense_m

    BML_REAL, intent(in) :: alpha
    type(bml_matrix_dense_BML_PRECISION_NAME_t), intent(inout) :: a
    BML_REAL, intent(in) :: beta
    type(bml_matrix_dense_BML_PRECISION_NAME_t), intent(in) :: b

    a%matrix = alpha*a%matrix+beta*b%matrix

  end subroutine add_two_dense_BML_PRECISION_NAME

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! \param alpha Factor \f$ \alpha \f$
  !! \param a Matrix A
  !! \param c Matrix C
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_two_dense_BML_PRECISION_NAME(alpha, a, c, beta)

    use bml_type_dense_m

    BML_REAL, intent(in) :: alpha
    type(bml_matrix_dense_BML_PRECISION_NAME_t), intent(in) :: a
    type(bml_matrix_dense_BML_PRECISION_NAME_t), intent(out) :: c
    BML_REAL, intent(in) :: beta

    integer :: i

    c%matrix = alpha*a%matrix
    do i = 1, a%n
       c%matrix(i, i) = c%matrix(i, i) + beta
    end do

  end subroutine add_identity_two_dense_BML_PRECISION_NAME

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! \param a Matrix A
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_self_dense_BML_PRECISION_NAME(a, alpha, beta)

    use bml_type_dense_m

    type(bml_matrix_dense_BML_PRECISION_NAME_t), intent(inout) :: a
    BML_REAL, intent(in) :: alpha
    BML_REAL, intent(in) :: beta

    integer :: i

    a%matrix = alpha*a%matrix
    do i = 1, a%N
       a%matrix(i, i) = a%matrix(i, i) + beta
    end do

  end subroutine add_identity_self_dense_BML_PRECISION_NAME

end module bml_add_dense_BML_PRECISION_NAME_m
