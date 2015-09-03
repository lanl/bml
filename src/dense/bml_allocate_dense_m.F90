!> \copyright Los Alamos National Laboratory 2015

!> Initialization of dense matrices.
module bml_allocate_dense_m
  implicit none
contains

  !> Allocate a dense matrix.
  !!
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param matrix_precision The precision of the matrix.
  subroutine allocate_matrix_dense(n, a, matrix_precision)

    use bml_type_m
    use bml_type_dense_m
    use bml_error_m

    integer, intent(in) :: n
    class(bml_matrix_t), pointer, intent(inout) :: a
    character(len=*), intent(in) :: matrix_precision

    if(associated(A)) then
       select type(A)
       class is(bml_matrix_dense_t)
          call deallocate_matrix_dense(A)
       class default
          call bml_error(__FILE__, __LINE__, "unknow matrix type")
       end select
    end if

    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       allocate(bml_matrix_dense_single_t::A)
    case(BML_PRECISION_DOUBLE)
       allocate(bml_matrix_dense_double_t::A)
    case default
       call bml_error(__FILE__, __LINE__, "unknown precision "//trim(matrix_precision))
    end select

    select type(A)
    type is(bml_matrix_dense_single_t)
       allocate(A%matrix(N, N))
       A%matrix = 0
    type is(bml_matrix_dense_double_t)
       allocate(A%matrix(N, N))
       A%matrix = 0
    class default
       call bml_error(__FILE__, __LINE__, "unknown type")
    end select

    A%N = N

  end subroutine allocate_matrix_dense

  !> Deallocate a dense matrix.
  !!
  !! @param A The matrix.
  subroutine deallocate_matrix_dense(A)

    use bml_type_dense_m
    use bml_error_m

    class(bml_matrix_dense_t), intent(inout) :: A

    select type(A)
    type is(bml_matrix_dense_single_t)
       deallocate(A%matrix)
    type is(bml_matrix_dense_double_t)
       deallocate(A%matrix)
    class default
       call bml_error(__FILE__, __LINE__, "unknown type")
    end select

  end subroutine deallocate_matrix_dense

  !> Initialize a dense random matrix.
  !!
  !! \param N The matrix size.
  !! \param A The matrix.
  !! \param matrix_precision The precision of the matrix.
  subroutine random_matrix_dense(N, A, matrix_precision)

    use bml_type_m
    use bml_type_dense_m
    use bml_error_m

    integer, intent(in) :: N
    class(bml_matrix_t), pointer, intent(inout) :: A
    character(len=*), intent(in) :: matrix_precision

    call allocate_matrix_dense(N, A, matrix_precision)
    select type(A)
    type is(bml_matrix_dense_single_t)
       call random_number(A%matrix)
    type is(bml_matrix_dense_double_t)
       call random_number(A%matrix)
    class default
       call bml_error(__FILE__, __LINE__, "unknown type")
    end select

  end subroutine random_matrix_dense

  !> Initialize a dense identity matrix.
  !!
  !! \param N The matrix size.
  !! \param A The matrix.
  !! \param matrix_precision The precision of the matrix.
  subroutine identity_matrix_dense(N, A, matrix_precision)

    use bml_type_m
    use bml_type_dense_m
    use bml_error_m

    integer, intent(in) :: N
    class(bml_matrix_t), pointer, intent(inout) :: A
    character(len=*), intent(in) :: matrix_precision

    integer :: i

    call allocate_matrix_dense(N, A, matrix_precision)
    select type(A)
    type is(bml_matrix_dense_single_t)
       do i = 1, N
          A%matrix(i, i) = 1
       end do
    type is(bml_matrix_dense_double_t)
       do i = 1, N
          A%matrix(i, i) = 1
       end do
    class default
       call bml_error(__FILE__, __LINE__, "unknown type")
    end select

  end subroutine identity_matrix_dense

end module bml_allocate_dense_m
