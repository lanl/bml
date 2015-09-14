module bml_convert

  implicit none

  private

  !> The interfaces to the C API.
  interface

     function bml_convert_from_dense_single_C(matrix_type, matrix_precision, n, a, threshold) &
          bind(C, name="bml_convert_from_dense_wrapper_single")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: n
       real(C_FLOAT), intent(in) :: a(:, :)
       real(C_DOUBLE), value, intent(in) :: threshold
       type(C_PTR) :: bml_convert_from_dense_single_C
     end function bml_convert_from_dense_single_C

     function bml_convert_from_dense_double_C(matrix_type, matrix_precision, n, a, threshold) &
          bind(C, name="bml_convert_from_dense_wrapper_double")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: n
       real(C_DOUBLE), intent(in) :: a(:, :)
       real(C_DOUBLE), value, intent(in) :: threshold
       type(C_PTR) :: bml_convert_from_dense_double_C
     end function bml_convert_from_dense_double_C

     function bml_convert_to_dense_C(a) bind(C, name="bml_convert_to_dense")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       type(C_PTR) :: bml_convert_to_dense_C
     end function bml_convert_to_dense_C

  end interface

  interface bml_convert_from_dense
     module procedure bml_convert_from_dense_single
     module procedure bml_convert_from_dense_double
  end interface bml_convert_from_dense

  interface bml_convert_to_dense
     module procedure bml_convert_to_dense_single
     module procedure bml_convert_to_dense_double
  end interface bml_convert_to_dense

  public :: bml_convert_from_dense
  public :: bml_convert_to_dense

contains

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param matrix_type The matrix type
  !! \param a_dense The dense matrix
  !! \param a The bml matrix
  !! \param threshold The matrix element magnited threshold
  subroutine bml_convert_from_dense_single(matrix_type, matrix_precision, a_dense, a, threshold)

    use bml_types
    use bml_interface

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    real, target, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    double precision, intent(in) :: threshold

    a%ptr = bml_convert_from_dense_single_C(get_enum_id(matrix_type), &
         get_enum_id(matrix_precision), &
         size(a_dense, 1), a_dense, threshold)

  end subroutine bml_convert_from_dense_single

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param matrix_type The matrix type
  !! \param a_dense The dense matrix
  !! \param a The bml matrix
  !! \param threshold The matrix element magnited threshold
  subroutine bml_convert_from_dense_double(matrix_type, matrix_precision, a_dense, a, threshold)

    use bml_types
    use bml_interface

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    double precision, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    double precision, intent(in) :: threshold

    a%ptr = bml_convert_from_dense_double_C(get_enum_id(matrix_type), &
         get_enum_id(matrix_precision), &
         size(a_dense, 1), a_dense, threshold)

  end subroutine bml_convert_from_dense_double

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_single(a, a_dense)

    use bml_types

    type(bml_matrix_t), intent(in) :: a
    real, allocatable, intent(out) :: a_dense(:, :)

    !allocate(a_dense(bml_get_size(a), bml_get_size(a)))
    !!call bml_convert_to_dense(a, a_dense)

  end subroutine bml_convert_to_dense_single

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_double(a, a_dense)

    use bml_types

    type(bml_matrix_t), intent(in) :: a
    double precision, allocatable, intent(out) :: a_dense(:, :)

  end subroutine bml_convert_to_dense_double

end module bml_convert
