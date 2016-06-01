!> Matrix allocation functions.
module bml_allocate_m

  use bml_c_interface_m
  use bml_types_m
  use bml_interface_m

  implicit none
  private

  public :: bml_banded_matrix
  public :: bml_identity_matrix
  public :: bml_random_matrix
  public :: bml_zero_matrix
  public :: bml_update_domain

contains

  !> Create the zero matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param matrix_type The matrix type.
  !! \param element_type  Element type of the matrix.
  !! \param element_precision The precision of the matrix elements.
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param m The extra arg.
  subroutine bml_zero_matrix(matrix_type, element_type, element_precision, &
      & n, m, a)

    character(len=*), intent(in) :: matrix_type, element_type
    integer, intent(in) :: element_precision
    integer(C_INT), intent(in) :: n, m
    type(bml_matrix_t), intent(inout) :: a

    call bml_deallocate(a)
    a%ptr = bml_zero_matrix_C(get_matrix_id(matrix_type), &
        & get_element_id(element_type, element_precision), n, m)

  end subroutine bml_zero_matrix

  !> Create a banded matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param matrix_type The matrix type.
  !! \param element_type  Element type of the matrix.
  !! \param element_precision The precision of the matrix.
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param m The extra arg.
  subroutine bml_banded_matrix(matrix_type, element_type, element_precision, &
      & n, m, a)

    character(len=*), intent(in) :: matrix_type, element_type
    integer, intent(in) :: element_precision
    integer(C_INT), intent(in) :: n, m
    type(bml_matrix_t), intent(inout) :: a

    call bml_deallocate(a)
    a%ptr = bml_banded_matrix_C(get_matrix_id(matrix_type), &
        & get_element_id(element_type, element_precision), n, m)

  end subroutine bml_banded_matrix

  !> Create a random matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param matrix_type The matrix type.
  !! \param element_type  Element type of the matrix.
  !! \param element_precision The precision of the matrix.
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param m The extra arg.
  subroutine bml_random_matrix(matrix_type, element_type, element_precision, &
      & n, m, a)

    character(len=*), intent(in) :: matrix_type, element_type
    integer, intent(in) :: element_precision
    integer(C_INT), intent(in) :: n, m
    type(bml_matrix_t), intent(inout) :: a

    call bml_deallocate(a)
    a%ptr = bml_random_matrix_C(get_matrix_id(matrix_type), &
        & get_element_id(element_type, element_precision), n, m)

  end subroutine bml_random_matrix

  !> Create the identity matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param matrix_type The matrix type.
  !! \param element_type  Element type of the matrix.
  !! \param element_precision The precision of the matrix.
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param m The extra arg.
  subroutine bml_identity_matrix(matrix_type, element_type, element_precision, &
      & n, m, a)

    character(len=*), intent(in) :: matrix_type, element_type
    integer, intent(in) :: element_precision
    integer(C_INT), intent(in) :: n, m
    type(bml_matrix_t), intent(inout) :: a

    call bml_deallocate(a)
    a%ptr = bml_identity_matrix_C(get_matrix_id(matrix_type), &
        & get_element_id(element_type, element_precision), n, m)

  end subroutine bml_identity_matrix

  !> Update domain of a matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param a The matrix.
  !! \param globalPartMin First part on each rank
  !! \param globalPartMax Last part on each rank
  !! \param nnodesInPart Number of nodes in each part
  subroutine bml_update_domain(a, globalPartMin, globalPartMax, nnodesInPart)

    integer(C_INT), target, intent(in) :: globalPartMin(*)
    integer(C_INT), target, intent(in) :: globalPartMax(*)
    integer(C_INT), target, intent(in) :: nnodesInPart(*)
    type(bml_matrix_t), intent(inout) :: a

    call bml_update_domain_C(a%ptr, c_loc(globalPartMin), c_loc(globalPartMax), &
        c_loc(nnodesInPart))

  end subroutine bml_update_domain

end module bml_allocate_m
