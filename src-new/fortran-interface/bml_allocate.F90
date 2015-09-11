module bml_allocate

  implicit none

  private

  !> The interfaces to the C API.
  interface
     subroutine bml_deallocate_C(a) bind(C, name="bml_deallocate")
       use, intrinsic :: iso_C_binding
       type(C_PTR) :: a
     end subroutine bml_deallocate_C

     function bml_zero_matrix_C(matrix_type, matrix_precision, n) bind(C, name="bml_allocate")
       use, intrinsic :: iso_C_binding
       integer(C_INT), intent(in) :: matrix_type
       integer(C_INT), intent(in) :: matrix_precision
       integer(C_INT), intent(in) :: n
       type(C_PTR) :: bml_zero_matrix_C
     end function bml_zero_matrix_C

  end interface

  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  integer, parameter :: bml_matrix_type_dense_enum_id = 0
  integer, parameter :: bml_matrix_precision_single_enum_id = 0
  integer, parameter :: bml_matrix_precision_double_enum_id = 1

  public :: bml_deallocate
  public :: bml_random_matrix
  public :: bml_identity_matrix
  public :: bml_zero_matrix

contains

  !> Convert the matrix type and precisions strings into enum values.
  function get_enum_id(type_string) result(id)

    use bml_types

    character(len=*), intent(in) :: type_string
    integer :: id

    select case(type_string)
    case(BML_PRECISION_SINGLE)
       id = bml_matrix_precision_single_enum_id
    case(BML_PRECISION_DOUBLE)
       id = bml_matrix_precision_double_enum_id
    case(BML_MATRIX_DENSE)
       id = bml_matrix_type_dense_enum_id
    case default
       print *, "unknown type string "//trim(type_string)
       error stop
    end select

  end function get_enum_id

  !> Deallocate a matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param a The matrix.
  subroutine bml_deallocate(a)
    use bml_types
    type(bml_matrix_t) :: a
    call bml_deallocate_C(a%ptr)
  end subroutine bml_deallocate

  !> Create the zero matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param matrix_type The matrix type.
  !! \param matrix_precision The precision of the matrix.
  !! \param n The matrix size.
  !! \param a The matrix.
  subroutine bml_zero_matrix(matrix_type, matrix_precision, n, a)

    use bml_types

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n
    type(bml_matrix_t), intent(inout) :: a

    a%ptr = bml_zero_matrix_C(get_enum_id(matrix_type), get_enum_id(matrix_precision), n)

  end subroutine bml_zero_matrix

  !> Create a random matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param matrix_type The matrix type.
  !! \param matrix_precision The precision of the matrix.
  !! \param n The matrix size.
  !! \param a The matrix.
  subroutine bml_random_matrix(matrix_type, matrix_precision, n, a)
    use bml_types
    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n
    type(bml_matrix_t), intent(inout) :: a
  end subroutine bml_random_matrix

  !> Create the identity matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param matrix_type The matrix type.
  !! \param matrix_precision The precision of the matrix.
  !! \param n The matrix size.
  !! \param a The matrix.
  subroutine bml_identity_matrix(matrix_type, matrix_precision, n, a)
    use bml_types
    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n
    type(bml_matrix_t), intent(inout) :: a
  end subroutine bml_identity_matrix

end module bml_allocate
