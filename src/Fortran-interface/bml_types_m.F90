!> The basic bml types.
module bml_types_m
  use, intrinsic :: iso_c_binding
  private

  public :: bml_vector_t, bml_matrix_t
  public :: bml_deallocate
  public :: BML_MATRIX_DENSE, BML_MATRIX_ELLPACK
  public :: BML_ELEMENT_REAL, BML_ELEMENT_COMPLEX

  !> The bml vector type.
  type :: bml_vector_t
     !> The C pointer to the vector.
     type(C_PTR) :: ptr = C_NULL_PTR
  contains
     final :: destruct_bml_vector_t
  end type bml_vector_t

  !> The bml matrix type.
  type :: bml_matrix_t
     !> The C pointer to the matrix.
     type(C_PTR) :: ptr = C_NULL_PTR
  contains
     final :: destruct_bml_matrix_t
  end type bml_matrix_t

  !> The bml-dense matrix type identifier.
  character(len=*), parameter :: BML_MATRIX_DENSE = "dense"

  !> The bml-ellpack matrix type identifier.
  character(len=*), parameter :: BML_MATRIX_ELLPACK = "ellpack"

  !> The single precision identifier.
  character(len=*), parameter :: BML_ELEMENT_REAL = "real"

  !> The double-precision identifier.
  character(len=*), parameter :: BML_ELEMENT_COMPLEX = "complex"


  interface
    subroutine bml_deallocate_C(a) bind(C, name="bml_deallocate")
      import :: C_PTR
      type(C_PTR) :: a
     end subroutine bml_deallocate_C
   end interface

contains

  !> Deallocate a matrix.
  !!
  !! \ingroup allocate_group_Fortran
  !!
  !! \param a The matrix.
  subroutine bml_deallocate(a)
    type(bml_matrix_t) :: a
    call bml_deallocate_C(a%ptr)
  end subroutine bml_deallocate


  subroutine destruct_bml_vector_t(this)
    type(bml_vector_t), intent(inout) :: this

    print *, "DESTRUCTOR for bml_vector not implemented yet."
    print *, "You possibly leak memory here."
    this%ptr = C_NULL_PTR
    
  end subroutine destruct_bml_vector_t


  subroutine destruct_bml_matrix_t(this)
    type(bml_matrix_t), intent(inout) :: this

    if (c_associated(this%ptr)) then
      call bml_deallocate(this)
    end if
    this%ptr = C_NULL_PTR

  end subroutine destruct_bml_matrix_t
    

end module bml_types_m
