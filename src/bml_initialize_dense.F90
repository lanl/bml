!> @copyright Los Alamos National Laboratory 2015

!> Initialization of dense matrices.
module bml_initialize_dense

  use bml_type_dense

  implicit none

contains

  !> Allocate a dense matrix.
  !!
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine allocate_matrix_dense(N, A)

    integer, intent(in) :: N
    type(matrix_dense_t), intent(inout) :: A

    A%N = N
    if(allocated(A%dense_matrix)) then
       deallocate(A%dense_matrix)
    endif
    allocate(A%dense_matrix(N, N))

  end subroutine allocate_matrix_dense

  !> Initialize a dense zero matrix.
  !!
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine zero_matrix_dense(N, A)

    integer, intent(in) :: N
    type(matrix_dense_t), intent(inout) :: A

    call allocate_matrix_dense(N, A)
    A%dense_matrix = 0

  end subroutine zero_matrix_dense

  !> Initialize a dense random matrix.
  !!
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine random_matrix_dense(N, A)

    integer, intent(in) :: N
    type(matrix_dense_t), intent(inout) :: A

    call allocate_matrix_dense(N, A)
    call random_number(A%dense_matrix)

  end subroutine random_matrix_dense

  !> Initialize a dense identity matrix.
  !!
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine identity_matrix_dense(N, A)

    integer, intent(in) :: N
    type(matrix_dense_t), intent(inout) :: A

    integer :: i

    call zero_matrix_dense(N, A)
    do i = 1, N
       A%dense_matrix(i, i) = 1
    enddo

  end subroutine identity_matrix_dense

end module bml_initialize_dense
