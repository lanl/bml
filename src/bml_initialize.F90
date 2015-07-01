!> @copyright Los Alamos National Laboratory 2015

!> Matrix initialization.
module bml_initialize

  use bml_initialize_dense

contains

  !> Allocate a matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine allocate_matrix(matrix_type, N, A)

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(matrix_t), allocatable, intent(inout) :: A

    select case(matrix_type)
    case("dense")
       allocate(matrix_dense_t::A)
       select type(A)
       type is(matrix_dense_t)
          call allocate_matrix_dense(N, A)
       end select
    case default
       write(*, *) "[initialize] unsupported matrix type"
       error stop
    end select

  end subroutine allocate_matrix

  !> Initialize a zero matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine zero_matrix(matrix_type, N, A)

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(matrix_t), allocatable, intent(inout) :: A

  end subroutine zero_matrix

  !> Initialize a random matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine random_matrix(matrix_type, N, A)

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(matrix_t), intent(inout) :: A

  end subroutine random_matrix

  !> Initialize a identity matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine identity_matrix(matrix_type, N, A)

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(matrix_t), intent(inout) :: A

  end subroutine identity_matrix

end module bml_initialize
