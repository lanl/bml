module bml_add_MATRIX_TYPE_m

  use bml_typed_m

  implicit none

  private

  !> \addtogroup add_group_Fortran
  !! @{

  !> Add two matrices.
  interface bml_add
     module procedure add_two_MATRIX_TYPE
     module procedure add_three_MATRIX_TYPE
  end interface bml_add

  !> Add identity matrix to a matrix.
  interface bml_add_identity
     module procedure add_identity_one_MATRIX_TYPE
     module procedure add_identity_two_MATRIX_TYPE
  end interface bml_add_identity
  !> @}

  public :: bml_add
  public :: bml_add_identity

contains

  !> Add two matrices.
  !!
  !! \f$ A \leftarrow \alpha A + \beta B \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param alpha Factor \f$ \alpha \f$
  !! \param a Matrix \f$ A \f$
  !! \param beta Factor \f$ \beta \f$
  !! \param b Matrix \f$ B \f$
  subroutine add_two_MATRIX_TYPE(alpha, a, beta, b)

    use bml_types_m

    REAL_TYPE, intent(in) :: alpha
    type(bml_matrix_t), intent(inout) :: a
    REAL_TYPE, intent(in) :: beta
    type(bml_matrix_t), intent(in) :: b

  end subroutine add_two_MATRIX_TYPE

  !> Add two matrices.
  !!
  !! \f$ C \leftarrow \alpha A + \beta B \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param alpha Factor \f$ \alpha \f$
  !! \param a Matrix \f$ A \f$
  !! \param beta Factor \f$ \beta \f$
  !! \param b Matrix \f$ B \f$
  !! \param c Matrix \f$ C \f$
  subroutine add_three_MATRIX_TYPE(alpha, a, beta, b, c)

    use bml_types_m

    REAL_TYPE, intent(in) :: alpha
    type(bml_matrix_t), intent(in) :: a
    REAL_TYPE, intent(in) :: beta
    type(bml_matrix_t), intent(in) :: b
    type(bml_matrix_t), intent(inout) :: c

  end subroutine add_three_MATRIX_TYPE

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ A \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param a Matrix A
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_one_MATRIX_TYPE(a, alpha, beta)

    use bml_types_m

    REAL_TYPE, intent(in) :: alpha
    type(bml_matrix_t), intent(inout) :: a
    REAL_TYPE, intent(in) :: beta

  end subroutine add_identity_one_MATRIX_TYPE

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ C \leftarrow \alpha A + \beta \mathrm{Id} \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param alpha Factor \f$ \alpha \f$
  !! \param a Matrix A
  !! \param c Matrix C
  !! \param beta Factor \f$ \beta \f$
  subroutine add_identity_two_MATRIX_TYPE(alpha, a, c, beta)

    use bml_types_m

    REAL_TYPE, intent(in) :: alpha
    type(bml_matrix_t), intent(in) :: a
    REAL_TYPE, intent(in) :: beta
    type(bml_matrix_t), intent(inout) :: c

  end subroutine add_identity_two_MATRIX_TYPE

end module bml_add_MATRIX_TYPE_m
