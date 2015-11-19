module bml_add_m

  implicit none

  private

  interface

     subroutine bml_add_C(a, b, alpha, beta, threshold) bind(C, name="bml_add")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       type(C_PTR), value, intent(in) :: b
       real(C_DOUBLE), value, intent(in) :: alpha
       real(C_DOUBLE), value, intent(in) :: beta
       real(C_DOUBLE), value, intent(in) :: threshold
     end subroutine bml_add_C

     subroutine bml_add_identity_C(a, beta, threshold) bind(C, name="bml_add_identity")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value :: a
       real(C_DOUBLE), value, intent(in) :: beta
       real(C_DOUBLE), value, intent(in) :: threshold
     end subroutine bml_add_identity_C

  end interface

  !> \addtogroup add_group_Fortran
  !! @{

  !> Add two matrices.
  interface bml_add
     module procedure add_two
  end interface bml_add

  !> Add identity matrix to a matrix.
  interface bml_add_identity
     module procedure add_identity_one
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
  !! \param threshold \f$ threshold \f$
  subroutine add_two(alpha, a, beta, b, threshold)

    use bml_types_m

    double precision, intent(in) :: alpha
    type(bml_matrix_t), intent(inout) :: a
    double precision, intent(in) :: beta
    type(bml_matrix_t), intent(in) :: b
    double precision, intent(in) :: threshold

    call bml_add_C(a%ptr, b%ptr, alpha, beta, threshold)

  end subroutine add_two

  !> Add a scaled identity matrix to a bml matrix.
  !!
  !! \f$ A \leftarrow A + \beta \mathrm{Id} \f$
  !!
  !! The optional scalars \f$ \alpha \f$ and \f$ \beta \f$ default to
  !! 1.
  !!
  !! \param a Matrix A
  !! \param alpha Factor \f$ \alpha \f$
  !! \param beta Factor \f$ \beta \f$
  !! \param threshold \f$ threshold \f$
  subroutine add_identity_one(a, alpha, threshold)

    use bml_types_m

    type(bml_matrix_t), intent(inout) :: a
    double precision, intent(in) :: alpha
    double precision, intent(in) :: threshold

    call bml_add_identity_C(a%ptr, alpha, threshold)

  end subroutine add_identity_one

end module bml_add_m
