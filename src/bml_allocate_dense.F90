!> @copyright Los Alamos National Laboratory 2015

!> Initialization of dense matrices.
module bml_allocate_dense

  use bml_type_dense

  implicit none

contains

  !> Allocate a dense matrix.
  !!
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine allocate_matrix_dense(N, A)

    integer, intent(in) :: N
    type(bml_matrix_dense_t), intent(inout) :: A

    A%N = N
    if(allocated(A%matrix)) then
       deallocate(A%matrix)
    endif
    allocate(A%matrix(N, N))

  end subroutine allocate_matrix_dense

  !> Deallocate a dense matrix.
  !!
  !! @param A The matrix.
  subroutine deallocate_matrix_dense(A)

    type(bml_matrix_dense_t), intent(inout) :: A

    deallocate(A%matrix)

  end subroutine deallocate_matrix_dense

  !> Initialize a dense zero matrix.
  !!
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine zero_matrix_dense(N, A)

    integer, intent(in) :: N
    type(bml_matrix_dense_t), intent(inout) :: A

    call allocate_matrix_dense(N, A)
    A%matrix = 0

  end subroutine zero_matrix_dense

  !> Initialize a dense random matrix.
  !!
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine random_matrix_dense(N, A)

    integer, intent(in) :: N
    type(bml_matrix_dense_t), intent(inout) :: A

    call allocate_matrix_dense(N, A)
    call random_number(A%matrix)

  end subroutine random_matrix_dense

  !> Initialize a dense identity matrix.
  !!
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine identity_matrix_dense(N, A)

    integer, intent(in) :: N
    type(bml_matrix_dense_t), intent(inout) :: A

    integer :: i

    call zero_matrix_dense(N, A)
    do i = 1, N
       A%matrix(i, i) = 1
    enddo

  end subroutine identity_matrix_dense

end module bml_allocate_dense
