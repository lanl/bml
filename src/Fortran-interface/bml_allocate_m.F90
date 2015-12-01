!> Matrix allocation functions.
module bml_allocate_m
  use, intrinsic :: iso_c_binding
  use bml_types_m
  use bml_interface_m
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


     function bml_zero_matrix_C(matrix_type, matrix_precision, n, m) bind(C, name="bml_zero_matrix")
       import :: C_INT, C_PTR
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: n
       integer(C_INT), value, intent(in) :: m
       type(C_PTR) :: bml_zero_matrix_C
     end function bml_zero_matrix_C

     function bml_banded_matrix_C(matrix_type, matrix_precision, n, m) bind(C, name="bml_banded_matrix")
       import :: C_INT, C_PTR
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: n
       integer(C_INT), value, intent(in) :: m
       type(C_PTR) :: bml_banded_matrix_C
     end function bml_banded_matrix_C

     function bml_random_matrix_C(matrix_type, matrix_precision, n, m) bind(C, name="bml_random_matrix")
       import :: C_INT, C_PTR
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: n
       integer(C_INT), value, intent(in) :: m
       type(C_PTR) :: bml_random_matrix_C
     end function bml_random_matrix_C

     function bml_identity_matrix_C(matrix_type, matrix_precision, n, m) bind(C, name="bml_identity_matrix")
       import :: C_INT, C_PTR
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: n
       integer(C_INT), value, intent(in) :: m
       type(C_PTR) :: bml_identity_matrix_C
     end function bml_identity_matrix_C

  end interface

  public :: bml_random_matrix
  public :: bml_banded_matrix
  public :: bml_identity_matrix
  public :: bml_zero_matrix

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
    type(bml_matrix_t), intent(out) :: a

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
    type(bml_matrix_t), intent(out) :: a

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
    type(bml_matrix_t), intent(out) :: a

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
    type(bml_matrix_t), intent(out) :: a

    a%ptr = bml_identity_matrix_C(get_matrix_id(matrix_type), &
        & get_element_id(element_type, element_precision), n, m)

  end subroutine bml_identity_matrix

end module bml_allocate_m
