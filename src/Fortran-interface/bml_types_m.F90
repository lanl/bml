!> The basic bml types.
module bml_types_m
  use bml_c_interface_m
  private

  public :: bml_vector_t, bml_matrix_t
  public :: bml_deallocate
  public :: BML_MATRIX_DENSE, BML_MATRIX_ELLPACK
  public :: BML_ELEMENT_REAL, BML_ELEMENT_COMPLEX

  ! NOTE: The object oriented approach using destructors, which would make
  ! explicit bml_deallocate() unnecessary and would prevent memory leaks, has
  ! been disabled, as the GNU Fortran compiler up to version 5.2 is not able
  ! create reliably working code. Once the reported bug
  !     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=68778
  ! is resolved, destructors can be enabled again, and the bml_deallocate()
  ! calls can vanish from the library and from user code as well. (BA)


  !> The bml vector type.
  type :: bml_vector_t
     !> The C pointer to the vector.
     type(C_PTR) :: ptr = C_NULL_PTR
  !contains
  !   procedure :: bml_vector_t_assign
  !   generic :: assignment(=) => bml_vector_t_assign
  !   final :: destruct_bml_vector_t
  end type bml_vector_t

  !> The bml matrix type.
  type :: bml_matrix_t
     !> The C pointer to the matrix.
     type(C_PTR) :: ptr = C_NULL_PTR
  !contains
  !   procedure :: bml_matrix_t_assign
  !   generic :: assignment(=) => bml_matrix_t_assign
  !   final :: destruct_bml_matrix_t
  end type bml_matrix_t

  !> The bml-dense matrix type identifier.
  character(len=*), parameter :: BML_MATRIX_DENSE = "dense"

  !> The bml-ellpack matrix type identifier.
  character(len=*), parameter :: BML_MATRIX_ELLPACK = "ellpack"

  !> The single precision identifier.
  character(len=*), parameter :: BML_ELEMENT_REAL = "real"

  !> The double-precision identifier.
  character(len=*), parameter :: BML_ELEMENT_COMPLEX = "complex"


contains

  !> Deallocate a matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param a The matrix.
  subroutine bml_deallocate(a)
    type(bml_matrix_t), intent(inout) :: a

    if (c_associated(a%ptr)) then
      call bml_deallocate_C(a%ptr)
    end if
    a%ptr = C_NULL_PTR

  end subroutine bml_deallocate


  !subroutine bml_vector_t_assign(this, other)
  !  class(bml_vector_t), intent(inout) :: this
  !  class(bml_vector_t), intent(in) :: other
  !
  !  print *, "Direct assignment between vectors not implemented"
  !  error stop
  !
  !end subroutine bml_vector_t_assign
  !
  !
  !subroutine destruct_bml_vector_t(this)
  !  type(bml_vector_t), intent(inout) :: this
  !
  !  print *, "DESTRUCTOR for bml_vector not implemented yet."
  !  print *, "You possibly leak memory here."
  !  this%ptr = C_NULL_PTR
  !
  !end subroutine destruct_bml_vector_t
  !
  !
  !subroutine bml_matrix_t_assign(this, other)
  !  class(bml_matrix_t), intent(out) :: this
  !  class(bml_matrix_t), intent(in) :: other
  !
  !  if (c_associated(other%ptr)) then
  !    this%ptr = bml_copy_new_C(other%ptr)
  !  end if
  !
  !end subroutine bml_matrix_t_assign
  !
  !
  !subroutine destruct_bml_matrix_t(this)
  !  type(bml_matrix_t), intent(inout) :: this
  !
  !    call bml_deallocate(this)
  !
  !end subroutine destruct_bml_matrix_t


end module bml_types_m
