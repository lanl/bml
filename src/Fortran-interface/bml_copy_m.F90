!> Copy operations on matrices.
module bml_copy_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_copy
  public :: bml_copy_new
  public :: bml_reorder
  public :: bml_save_domain
  public :: bml_restore_domain

contains

  !> Copy a matrix - result is a new matrix.
  !!
  !! \ingroup copy_group_F
  !!
  !! \param a Matrix to copy
  !! \param b The copy
  subroutine bml_copy_new(a, b)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: b

    call bml_deallocate(b)
    b%ptr = bml_copy_new_C(a%ptr)

  end subroutine bml_copy_new

  !> Copy a matrix - result is a new matrix.
  !!
  !! \ingroup copy_group_F
  !!
  !! \param a Matrix to copy
  !! \param b The copy
  subroutine bml_copy(a, b)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: b

    call bml_copy_C(a%ptr, b%ptr)

  end subroutine bml_copy

  !> Reorder a matrix.
  !!
  !! \ingroup copy_group_F
  !!
  !! \param a Matrix to reorder
  !! \param perm The permutation matrix
  subroutine bml_reorder(a, perm)

    use bml_types_m

    type(bml_matrix_t), intent(inout) :: a
    integer(C_INT), target, intent(in) :: perm(*)

    call bml_reorder_C(a%ptr, c_loc(perm))

  end subroutine bml_reorder


  !> Save the matrix's domain.
  !!
  !! \ingroup copy_group_F
  !!
  !! \param a Matrix's domain to save
  subroutine bml_save_domain(a)

    use bml_types_m

    type(bml_matrix_t), intent(inout) :: a

    call bml_save_domain_C(a%ptr)

  end subroutine bml_save_domain

  !> Restore the matrix's domain.
  !!
  !! \ingroup copy_group_F
  !!
  !! \param a Matrix's domain to restore
  subroutine bml_restore_domain(a)

    use bml_types_m

    type(bml_matrix_t), intent(inout) :: a

    call bml_restore_domain_C(a%ptr)

  end subroutine bml_restore_domain

end module bml_copy_m
