!> \copyright Los Alamos National Laboratory 2015

!> Some format conversion functions.
module bml_convert_m
  implicit none
contains

  !> Convert a matrix into a dense matrix.
  !!
  !! \param A The bml matrix
  !! \param A_dense The dense matrix
  subroutine convert_to_dense(A, A_dense)

    use bml_type_dense

    use bml_convert_dense
    use bml_error_m

    class(bml_matrix_t), allocatable, intent(in) :: A
    double precision, allocatable, intent(out) :: A_dense(:, :)

    if(.not. allocated(A)) then
       call warning(__FILE__, __LINE__, "A is not allocated")
    else
       select type(A)
       type is(bml_matrix_dense_t)
          call convert_to_dense_dense(A, A_dense)
       class default
          call error(__FILE__, __LINE__, "unknown matrix type")
       end select
    endif

  end subroutine convert_to_dense

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param matrix_type The matrix type
  !! \param A_dense The dense matrix
  !! \param A The bml matrix
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense(matrix_type, A_dense, A, threshold)

    use bml_type_dense

    use bml_allocate_m
    use bml_convert_dense
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    double precision, intent(in) :: A_dense(:, :)
    class(bml_matrix_t), allocatable, intent(out) :: A
    double precision, optional, intent(in) :: threshold

    if(size(A_dense, 1) /= size(A_dense, 2)) then
       call error(__FILE__, __LINE__, "[FIXME] only square matrices")
    endif

    call allocate_matrix(matrix_type, size(A_dense, 1), A)

    select type(A)
    type is(bml_matrix_dense_t)
       call convert_from_dense_dense(A_dense, A, threshold)
    class default
       call error(__FILE__, __LINE__, "unknonw matrix type")
    end select

  end subroutine convert_from_dense

end module bml_convert_m
