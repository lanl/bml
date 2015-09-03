!> \copyright Los Alamos National Laboratory 2015

!> Some format conversion functions.
module bml_convert_m

  implicit none

  private

  !> Convert from bml to dense matrix.
  interface bml_convert_to_dense
     module procedure :: convert_to_dense_single
     module procedure :: convert_to_dense_double
  end interface bml_convert_to_dense

  !> Convert from dense matrix to bml.
  interface bml_convert_from_dense
     module procedure :: convert_from_dense_single
     module procedure :: convert_from_dense_double
  end interface bml_convert_from_dense

  public :: bml_convert_to_dense
  public :: bml_convert_from_dense

contains

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param A The bml matrix
  !! \param A_dense The dense matrix
  subroutine convert_to_dense_single(A, A_dense)

    use bml_type_m
    use bml_type_dense_m
    use bml_type_ellpack_m
    use bml_convert_dense_m
    use bml_convert_ellpack_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: A
    real, allocatable, intent(out) :: A_dense(:, :)

    select type(A)
    type is(bml_matrix_dense_single_t)
       call convert_to_dense_dense(A, A_dense)
    type is(bml_matrix_ellpack_single_t)
       call convert_to_dense_ellpack(A, A_dense)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine convert_to_dense_single

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param A The bml matrix
  !! \param A_dense The dense matrix
  subroutine convert_to_dense_double(A, A_dense)

    use bml_type_m
    use bml_type_dense_m
    use bml_type_ellpack_m
    use bml_convert_dense_m
    use bml_convert_ellpack_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: A
    double precision, allocatable, intent(out) :: A_dense(:, :)

    select type(A)
    type is(bml_matrix_dense_double_t)
       call convert_to_dense_dense(A, A_dense)
    type is(bml_matrix_ellpack_double_t)
       call convert_to_dense_ellpack(A, A_dense)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine convert_to_dense_double

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param matrix_type The matrix type
  !! \param A_dense The dense matrix
  !! \param A The bml matrix
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_single(matrix_type, A_dense, A, threshold)

    use bml_type_m
    use bml_type_dense_m
    use bml_type_ellpack_m
    use bml_allocate_m
    use bml_convert_dense_m
    use bml_convert_ellpack_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    real, intent(in) :: A_dense(:, :)
    class(bml_matrix_t), allocatable, intent(out) :: A
    real, optional, intent(in) :: threshold

    if(size(A_dense, 1) /= size(A_dense, 2)) then
       call bml_error(__FILE__, __LINE__, "[FIXME] only square matrices")
    end if

    call bml_allocate(matrix_type, size(A_dense, 1), A, BML_PRECISION_SINGLE)

    select type(A)
    type is(bml_matrix_dense_single_t)
       call convert_from_dense_dense(A_dense, A, threshold)
    type is(bml_matrix_ellpack_single_t)
       call convert_from_dense_ellpack(A_dense, A, threshold)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine convert_from_dense_single

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param matrix_type The matrix type
  !! \param A_dense The dense matrix
  !! \param A The bml matrix
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_double(matrix_type, A_dense, A, threshold)

    use bml_type_m
    use bml_type_dense_m
    use bml_allocate_m
    use bml_convert_dense_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    double precision, intent(in) :: A_dense(:, :)
    class(bml_matrix_t), allocatable, intent(out) :: A
    double precision, optional, intent(in) :: threshold

    if(size(A_dense, 1) /= size(A_dense, 2)) then
       call bml_error(__FILE__, __LINE__, "[FIXME] only square matrices")
    end if

    call bml_allocate(matrix_type, size(A_dense, 1), A, BML_PRECISION_DOUBLE)

    select type(A)
    type is(bml_matrix_dense_double_t)
       call convert_from_dense_dense(A_dense, A, threshold)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine convert_from_dense_double

end module bml_convert_m
