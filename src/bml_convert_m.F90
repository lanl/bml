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
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine convert_to_dense_single(a, a_dense)

    use bml_type_m
    use bml_type_dense_m
    use bml_type_ellpack_m
    use bml_convert_dense_m
    use bml_convert_ellpack_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: a
    real, allocatable, intent(out) :: a_dense(:, :)

    select type(a)
    type is(bml_matrix_dense_single_t)
       call convert_to_dense_dense(a, a_dense)
    type is(bml_matrix_ellpack_single_t)
       call convert_to_dense_ellpack(a, a_dense)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine convert_to_dense_single

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine convert_to_dense_double(a, a_dense)

    use bml_type_m
    use bml_type_dense_m
    use bml_type_ellpack_m
    use bml_convert_dense_m
    use bml_convert_ellpack_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: a
    double precision, allocatable, intent(out) :: a_dense(:, :)

    select type(a)
    type is(bml_matrix_dense_double_t)
       call convert_to_dense_dense(a, a_dense)
    type is(bml_matrix_ellpack_double_t)
       call convert_to_dense_ellpack(a, a_dense)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine convert_to_dense_double

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param matrix_type The matrix type
  !! \param a_dense The dense matrix
  !! \param a The bml matrix
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_single(matrix_type, a_dense, a, threshold)

    use bml_type_m
    use bml_type_dense_m
    use bml_type_ellpack_m
    use bml_allocate_m
    use bml_convert_dense_m
    use bml_convert_ellpack_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    real, intent(in) :: a_dense(:, :)
    class(bml_matrix_t), allocatable, intent(out) :: a
    real, optional, intent(in) :: threshold

    if(size(a_dense, 1) /= size(a_dense, 2)) then
       call bml_error(__FILE__, __LINE__, "[FIXME] only square matrices")
    end if

    call bml_allocate(matrix_type, size(a_dense, 1), a, BML_PRECISION_SINGLE)

    select type(a)
    type is(bml_matrix_dense_single_t)
       call convert_from_dense_dense(a_dense, a, threshold)
    type is(bml_matrix_ellpack_single_t)
       call convert_from_dense_ellpack(a_dense, a, threshold)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine convert_from_dense_single

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param matrix_type The matrix type
  !! \param a_dense The dense matrix
  !! \param a The bml matrix
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_double(matrix_type, a_dense, a, threshold)

    use bml_type_m
    use bml_type_dense_m
    use bml_type_ellpack_m
    use bml_allocate_m
    use bml_convert_dense_m
    use bml_convert_ellpack_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    double precision, intent(in) :: a_dense(:, :)
    class(bml_matrix_t), allocatable, intent(out) :: a
    double precision, optional, intent(in) :: threshold

    if(size(a_dense, 1) /= size(a_dense, 2)) then
       call bml_error(__FILE__, __LINE__, "[FIXME] only square matrices")
    end if

    call bml_allocate(matrix_type, size(a_dense, 1), a, BML_PRECISION_DOUBLE)

    select type(a)
    type is(bml_matrix_dense_double_t)
       call convert_from_dense_dense(a_dense, a, threshold)
    type is(bml_matrix_ellpack_double_t)
       call convert_from_dense_ellpack(a_dense, a, threshold)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end subroutine convert_from_dense_double

end module bml_convert_m
