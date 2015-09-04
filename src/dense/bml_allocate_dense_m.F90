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
    class(bml_matrix_t), pointer, intent(out) :: a
    character(len=*), intent(in) :: matrix_precision

    !> if(associated(a)) then
    !>    select type(a)
    !>    class is(bml_matrix_dense_t)
    !>       call deallocate_matrix_dense(a)
    !>    class default
    !>       call bml_error(__FILE__, __LINE__, "unknow matrix type")
    !>    end select
    !> end if

    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       allocate(bml_matrix_dense_single_t::a)
    case(BML_PRECISION_DOUBLE)
       allocate(bml_matrix_dense_double_t::a)
    case default
       call bml_error(__FILE__, __LINE__, "unknown precision "//trim(matrix_precision))
    end select

    select type(a)
    type is(bml_matrix_dense_single_t)
       allocate(a%matrix(n, n))
       a%matrix = 0
    type is(bml_matrix_dense_double_t)
       allocate(a%matrix(n, n))
       a%matrix = 0
    class default
       call bml_error(__FILE__, __LINE__, "unknown type")
    end select

    a%n = n

  end subroutine allocate_matrix_dense

  !> Deallocate a dense matrix.
  !!
  !! @param a The matrix.
  subroutine deallocate_matrix_dense(a)

    use bml_type_dense_m
    use bml_error_m

    class(bml_matrix_dense_t), intent(inout) :: a

    select type(a)
    type is(bml_matrix_dense_single_t)
       deallocate(a%matrix)
    type is(bml_matrix_dense_double_t)
       deallocate(a%matrix)
    class default
       call bml_error(__FILE__, __LINE__, "unknown type")
    end select

  end subroutine deallocate_matrix_dense

  !> Initialize a dense random matrix.
  !!
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param matrix_precision The precision of the matrix.
  subroutine random_matrix_dense(n, a, matrix_precision)

    use bml_type_m
    use bml_type_dense_m
    use bml_error_m

    integer, intent(in) :: n
    class(bml_matrix_t), pointer, intent(out) :: a
    character(len=*), intent(in) :: matrix_precision

    call allocate_matrix_dense(n, a, matrix_precision)
    select type(a)
    type is(bml_matrix_dense_single_t)
       call random_number(a%matrix)
    type is(bml_matrix_dense_double_t)
       call random_number(a%matrix)
    class default
       call bml_error(__FILE__, __LINE__, "unknown type")
    end select

  end subroutine random_matrix_dense

  !> Initialize a dense identity matrix.
  !!
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param matrix_precision The precision of the matrix.
  subroutine identity_matrix_dense(n, a, matrix_precision)

    use bml_type_m
    use bml_type_dense_m
    use bml_error_m

    integer, intent(in) :: n
    class(bml_matrix_t), pointer, intent(out) :: a
    character(len=*), intent(in) :: matrix_precision

    integer :: i

    call allocate_matrix_dense(n, a, matrix_precision)
    select type(a)
    type is(bml_matrix_dense_single_t)
       do i = 1, n
          a%matrix(i, i) = 1
       end do
    type is(bml_matrix_dense_double_t)
       do i = 1, n
          a%matrix(i, i) = 1
       end do
    class default
       call bml_error(__FILE__, __LINE__, "unknown type")
    end select

  end subroutine identity_matrix_dense

end module bml_allocate_dense_m
