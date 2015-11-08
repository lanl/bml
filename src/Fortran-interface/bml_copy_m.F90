!> Copy operations on matrices.
module bml_copy_m

  implicit none

  private

  interface

     function bml_copy_new_C(a) bind(C, name="bml_copy_new")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       type(C_PTR) :: bml_copy_new_C
     end function bml_copy_new_C

     subroutine bml_copy_C(a, b) bind(C, name="bml_copy")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       type(C_PTR), value, intent(in) :: b
     end subroutine bml_copy_C

  end interface

  public :: bml_copy
  public :: bml_copy_new

contains

  !> Copy a matrix - result is a new matrix.
  !!
  !! \ingroup copy_group_F
  !!
  !! \param A Matrix to copy
  !! \return  A Copy of A
  function bml_copy_new(a) result(b)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t) :: b

    b%ptr = bml_copy_new_C(a%ptr)

  end function bml_copy_new

  !> Copy (assign) a matrix to another one.
  !!
  !! This operation performs \f$ B \leftarrow A \f$.
  !!
  !! \param a Matrix to copy.
  !! \param b Matrix to copy to.
  subroutine bml_copy(a, b)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: b

    call bml_copy_C(a%ptr, b%ptr)

  end subroutine bml_copy

end module bml_copy_m
