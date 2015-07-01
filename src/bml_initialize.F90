!> @copyright Los Alamos National Laboratory 2015

!> Matrix initialization.
module bml_initialize

  use bml_error
  use bml_initialize_dense
  use bml_type_dense

contains

  !> Allocate a matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine allocate_matrix(matrix_type, N, A)

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(matrix_t), allocatable, intent(out) :: A

    select case(matrix_type)
    case("dense")
       allocate(matrix_dense_t::A)
       select type(A)
       type is(matrix_dense_t)
          call allocate_matrix_dense(N, A)
       end select
    case default
       call error(__FILE__, __LINE__, "unsupported matrix type")
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
    class(matrix_t), allocatable, intent(out) :: A

    select case(matrix_type)
    case("dense")
       allocate(matrix_dense_t::A)
       select type(A)
       type is(matrix_dense_t)
          call zero_matrix_dense(N, A)
       end select
    case default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine zero_matrix

  !> Initialize a random matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine random_matrix(matrix_type, N, A)

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(matrix_t), allocatable, intent(out) :: A

    select case(matrix_type)
    case("dense")
       allocate(matrix_dense_t::A)
       select type(A)
       type is(matrix_dense_t)
          call random_matrix_dense(N, A)
       end select
    case default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine random_matrix

  !> Initialize a identity matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine identity_matrix(matrix_type, N, A)

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(matrix_t), allocatable, intent(out) :: A

    select case(matrix_type)
    case("dense")
       allocate(matrix_dense_t::A)
       select type(A)
       type is(matrix_dense_t)
          call identity_matrix_dense(N, A)
       end select
    case default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine identity_matrix

end module bml_initialize
