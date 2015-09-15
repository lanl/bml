!> Introspection procedures.
module bml_introspection

  implicit none

  interface
     !> Return the matrix size.
     function bml_get_size_C(a) result(n) bind(C, name="bml_get_size")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT) :: n
     end function bml_get_size_C
  end interface

contains

  !> Return the matrix size.
  !!
  !!\param a The matrix.
  !!\return The matrix size.
  function bml_get_size(a)

    use bml_types

    type(bml_matrix_t), intent(in) :: a
    integer :: bml_get_size

    bml_get_size = bml_get_size_C(a%ptr)

  end function bml_get_size

end module bml_introspection
