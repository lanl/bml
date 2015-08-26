!> \copyright Los Alamos National Laboratory 2015

!> Initialization of dense matrices.
module bml_allocate_ellpack_m
  implicit none
contains

  !> Allocate a sparse matrix.
  !!
  !! \param n The matrix size.
  !! \param a The matrix.
  !! \param matrix_precision The precision of the matrix.
  subroutine allocate_matrix_ellpack(n, a, matrix_precision)

    use bml_type_ellpack_m
    use bml_error_m

    integer, intent(in) :: n
    class(bml_matrix_t), allocatable, intent(inout) :: a
    character(len=*), intent(in) :: matrix_precision

    if(allocated(a)) then
       select type(a)
       class is(bml_matrix_ellpack_t)
          call deallocate_matrix_ellpack(a)
       class default
          call error(__FILE__, __LINE__, "unknow matrix type")
       end select
    end if

    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       allocate(bml_matrix_ellpack_single_t::a)
    case(BML_PRECISION_DOUBLE)
       allocate(bml_matrix_ellpack_double_t::a)
    case default
       call error(__FILE__, __LINE__, "unknown precision")
    end select

    select type(a)
    class is(bml_matrix_ellpack_t)
       allocate(a%number_entries(n))
       allocate(a%column_index(n, ELLPACK_M))
    class default
       call error(__FILE__, __LINE__, "unknown type")
    end select

    select type(a)
    type is(bml_matrix_ellpack_single_t)
       allocate(a%matrix(n, ELLPACK_M))
       a%matrix = 0
    type is(bml_matrix_ellpack_double_t)
       allocate(a%matrix(n, ELLPACK_M))
       a%matrix = 0
       class default
       call error(__FILE__, __LINE__, "unknown type")
    end select

    a%n = n

  end subroutine allocate_matrix_ellpack

  !> Deallocate a sparse matrix.
  !!
  !! @param a The matrix.
  subroutine deallocate_matrix_ellpack(a)

    use bml_type_ellpack_m
    use bml_error_m

    class(bml_matrix_ellpack_t), intent(inout) :: a

    select type(a)
    type is(bml_matrix_ellpack_single_t)
       deallocate(a%number_entries)
       deallocate(a%column_index)
       deallocate(a%matrix)
    type is(bml_matrix_ellpack_double_t)
       deallocate(a%number_entries)
       deallocate(a%column_index)
       deallocate(a%matrix)
    class default
       call error(__FILE__, __LINE__, "unknown type")
    end select

  end subroutine deallocate_matrix_ellpack

end module bml_allocate_ellpack_m
