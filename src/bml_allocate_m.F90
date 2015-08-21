!> \copyright Los Alamos National Laboratory 2015

!> Matrix initialization.
module bml_allocate_m
  implicit none
contains

  !> \addtogroup allocate_group
  !! @{

  !> Allocate a matrix.
  !!
  !! \param matrix_type The matrix type.
  !! \param N The matrix size.
  !! \param A The matrix.
  subroutine allocate_matrix(matrix_type, N, A)

    use bml_type_dense_m
    use bml_allocate_dense_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(bml_matrix_t), allocatable, intent(out) :: A

    select case(matrix_type)
    case(MATRIX_TYPE_NAME_DENSE_DOUBLE)
       allocate(bml_matrix_dense_t::A)
       select type(A)
       type is(bml_matrix_dense_t)
          call allocate_matrix_dense(N, A)
       end select
    case default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine allocate_matrix
  !> @}

  !> \addtogroup allocate_group
  !! @{

  !> Deallocate a matrix.
  !!
  !! @param A The matrix.
  subroutine deallocate_matrix(A)

    use bml_type_dense_m
    use bml_allocate_dense_m
    use bml_error_m

    class(bml_matrix_t), allocatable, intent(inout) :: A

    if(allocated(A)) then
       select type(A)
       type is(bml_matrix_dense_t)
          call deallocate_matrix_dense(A)
       class default
          call error(__FILE__, __LINE__, "unsupported matrix type")
       end select
       deallocate(A)
    endif

  end subroutine deallocate_matrix
  !> @}

  !> \addtogroup initialize_group
  !! @{

  !> Initialize a zero matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine zero_matrix(matrix_type, N, A)

    use bml_type_dense_m
    use bml_allocate_dense_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(bml_matrix_t), allocatable, intent(out) :: A

    select case(matrix_type)
    case(MATRIX_TYPE_NAME_DENSE_DOUBLE)
       call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, N, A)
       select type(A)
       type is(bml_matrix_dense_t)
          call zero_matrix_dense(N, A)
       end select
    case default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine zero_matrix
  !> @}

  !> \addtogroup initialize_group
  !! @{

  !> Initialize a random matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine random_matrix(matrix_type, N, A)

    use bml_type_dense_m
    use bml_allocate_dense_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(bml_matrix_t), allocatable, intent(out) :: A

    select case(matrix_type)
    case(MATRIX_TYPE_NAME_DENSE_DOUBLE)
       call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, N, A)
       select type(A)
       type is(bml_matrix_dense_t)
          call random_matrix_dense(N, A)
       end select
    case default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine random_matrix
  !> @}

  !> \addtogroup initialize_group
  !! @{

  !> Initialize a identity matrix.
  !!
  !! @param matrix_type The matrix type.
  !! @param N The matrix size.
  !! @param A The matrix.
  subroutine identity_matrix(matrix_type, N, A)

    use bml_type_dense_m
    use bml_allocate_dense_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(bml_matrix_t), allocatable, intent(out) :: A

    select case(matrix_type)
    case(MATRIX_TYPE_NAME_DENSE_DOUBLE)
       call allocate_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, N, A)
       select type(A)
       type is(bml_matrix_dense_t)
          call identity_matrix_dense(N, A)
       end select
    case default
       call error(__FILE__, __LINE__, "unsupported matrix type")
    end select

  end subroutine identity_matrix
  !> @}

end module bml_allocate_m
