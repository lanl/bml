!> \copyright Los Alamos National Laboratory 2015

!> Initialization of dense matrices.
module bml_allocate_ellpack_m
  implicit none
contains

  !> Allocate a sparse matrix.
  !!
  !! \param N The matrix size.
  !! \param A The matrix.
  !! \param matrix_precision The precision of the matrix.
  subroutine allocate_matrix_ellpack(N, A, matrix_precision)

    use bml_type_ellpack_m
    use bml_error_m

    integer, intent(in) :: N
    class(bml_matrix_t), allocatable, intent(inout) :: A
    character(len=*), intent(in) :: matrix_precision

    A%N = N

  end subroutine allocate_matrix_ellpack

  !> Deallocate a sparse matrix.
  !!
  !! @param A The matrix.
  subroutine deallocate_matrix_ellpack(A)

    use bml_type_ellpack_m
    use bml_error_m

    class(bml_matrix_ellpack_t), intent(inout) :: A

    select type(A)
    type is(bml_matrix_ellpack_single_t)
    type is(bml_matrix_ellpack_double_t)
    class default
       call error(__FILE__, __LINE__, "unknown type")
    end select

  end subroutine deallocate_matrix_ellpack

end module bml_allocate_ellpack_m
