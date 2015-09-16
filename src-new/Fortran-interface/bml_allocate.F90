!> Matrix allocation functions.
module bml_allocate

  implicit none

  private

  ! Note: According to Sec. 15.3.7.2.6: "any dummy argument without
  ! the value attribute corresponds to a formal parameter of the
  ! prototype that is of a pointer type, and the dummy argument is
  ! interoperable with an entity of the referenced type (ISO/IEC
  ! 9899:1999, 6.2.5, 7.17, and 7.18.1) of the formal parameter, ..."
  !
  ! In other words, a type(C_PTR) dummy argument is interoperable with
  ! the void** type.

  !> The interfaces to the C API.
  interface
     subroutine bml_deallocate_C(a) bind(C, name="bml_deallocate")
       use, intrinsic :: iso_C_binding
       type(C_PTR) :: a
     end subroutine bml_deallocate_C

     function bml_zero_matrix_C(matrix_type, matrix_precision, n, m) bind(C, name="bml_zero_matrix")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: n
       integer(C_INT), value, intent(in) :: m
       type(C_PTR) :: bml_zero_matrix_C
     end function bml_zero_matrix_C

     function bml_random_matrix_C(matrix_type, matrix_precision, n, m) bind(C, name="bml_random_matrix")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: n
       integer(C_INT), value, intent(in) :: m
       type(C_PTR) :: bml_random_matrix_C
     end function bml_random_matrix_C

     function bml_identity_matrix_C(matrix_type, matrix_precision, n, m) bind(C, name="bml_identity_matrix")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: n
       integer(C_INT), value, intent(in) :: m
       type(C_PTR) :: bml_identity_matrix_C
     end function bml_identity_matrix_C

  end interface

  public :: bml_deallocate
  public :: bml_random_matrix
  public :: bml_identity_matrix
  public :: bml_zero_matrix

contains

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
  !! \param m The extra arg.
  subroutine bml_zero_matrix(matrix_type, matrix_precision, n, a, m)

    use bml_types
    use bml_interface

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    type(bml_matrix_t), intent(inout) :: a

    if(c_associated(a%ptr)) then
       call bml_deallocate_C(a%ptr)
    end if
    a%ptr = bml_zero_matrix_C(get_enum_id(matrix_type), get_enum_id(matrix_precision), n, m)

  end subroutine bml_zero_matrix

  !> Create a random matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param matrix_type The matrix type.
  !! \param matrix_precision The precision of the matrix.
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param m The extra arg.
  subroutine bml_random_matrix(matrix_type, matrix_precision, n, a, m)

    use bml_types
    use bml_interface

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    type(bml_matrix_t), intent(inout) :: a

    if(c_associated(a%ptr)) then
       call bml_deallocate_C(a%ptr)
    end if
    a%ptr = bml_random_matrix_C(get_enum_id(matrix_type), get_enum_id(matrix_precision), n, m)

  end subroutine bml_random_matrix

  !> Create the identity matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param matrix_type The matrix type.
  !! \param matrix_precision The precision of the matrix.
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param m The extra arg.
  subroutine bml_identity_matrix(matrix_type, matrix_precision, n, a, m)

    use bml_types
    use bml_interface

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n
    type(bml_matrix_t), intent(inout) :: a

    if(c_associated(a%ptr)) then
       call bml_deallocate_C(a%ptr)
    end if
    a%ptr = bml_identity_matrix_C(get_enum_id(matrix_type), get_enum_id(matrix_precision), n, m)

  end subroutine bml_identity_matrix

end module bml_allocate
