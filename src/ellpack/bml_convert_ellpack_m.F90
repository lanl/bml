!> \copyright Los Alamos National Laboratory 2015

!> Some format conversion functions.
module bml_convert_ellpack_m

  implicit none

  private

  !> Convert bml to dense matrix.
  interface bml_convert_to_dense_ellpack
     module procedure convert_to_dense_ellpack_single
     module procedure convert_to_dense_ellpack_double
  end interface bml_convert_to_dense_ellpack

  !> Convert bml to dense matrix.
  interface bml_convert_from_dense_ellpack
     module procedure convert_from_dense_ellpack_single
     module procedure convert_from_dense_ellpack_double
  end interface bml_convert_from_dense_ellpack

  !> Utility function for getting the maximum bandwidth.
  interface get_max_bandwidth
     module procedure get_max_bandwidth_single
     module procedure get_max_bandwidth_double
  end interface get_max_bandwidth

  public :: bml_convert_to_dense_ellpack
  public :: bml_convert_from_dense_ellpack

contains

  !> Convert a matrix into a dense matrix.
  !!
  !! \f$ A \leftarrow A_{d} \f$
  !!
  !! \param a The bml matrix.
  !! \param a_dense The dense matrix.
  subroutine convert_to_dense_ellpack_single(a, a_dense)

    use bml_type_ellpack_m

    type(bml_matrix_ellpack_single_t), intent(in) :: a
    real, allocatable, intent(out) :: a_dense(:, :)

    integer :: i, j

    allocate(a_dense(a%n, a%n))
    a_dense = 0

    do i = 1, a%n
       do j = 1, a%number_entries(i)
          a_dense(i, a%column_index(i, j)) = a%matrix(i, j)
       end do
    end do

  end subroutine convert_to_dense_ellpack_single

  !> Convert a matrix into a dense matrix.
  !!
  !! \f$ A \leftarrow A_{d} \f$
  !!
  !! \param a The bml matrix.
  !! \param a_dense The dense matrix.
  subroutine convert_to_dense_ellpack_double(a, a_dense)

    use bml_type_ellpack_m

    type(bml_matrix_ellpack_double_t), intent(in) :: a
    double precision, allocatable, intent(out) :: a_dense(:, :)

    integer :: i, j

    allocate(a_dense(a%n, a%n))
    a_dense = 0

    do i = 1, a%n
       do j = 1, a%number_entries(i)
          a_dense(i, a%column_index(i, j)) = a%matrix(i, j)
       end do
    end do

  end subroutine convert_to_dense_ellpack_double

  !> Get the maximum bandwidth of a dense matrix.
  !!
  !! @param a The dense matrix.
  !! @return The maximum bandwidth.
  function get_max_bandwidth_single(a) result(max_m)

    real, intent(in) :: a(:, :)
    integer :: max_m

    integer :: i, j, max_m_i

    max_m = 0
    do i = 1, size(a, 1)
       max_m_i = 0
       do j = 1, size(a, 2)
          if(a(i, j) /= 0) then
             max_m_i = max_m_i+1
          end if
       end do
       max_m = max(max_m, max_m_i)
    end do

  end function get_max_bandwidth_single

  !> Get the maximum bandwidth of a dense matrix.
  !!
  !! @param a The dense matrix.
  !! @return The maximum bandwidth.
  function get_max_bandwidth_double(a) result(max_m)

    double precision, intent(in) :: a(:, :)
    integer :: max_m

    integer :: i, j, max_m_i

    max_m = 0
    do i = 1, size(a, 1)
       max_m_i = 0
       do j = 1, size(a, 2)
          if(a(i, j) /= 0) then
             max_m_i = max_m_i+1
          end if
       end do
       max_m = max(max_m, max_m_i)
    end do

  end function get_max_bandwidth_double

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param a_dense The dense matrix.
  !! \param a The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_ellpack_single(a_dense, a, threshold)

    use bml_type_ellpack_m
    use bml_error_m

    real, intent(in) :: a_dense(:, :)
    type(bml_matrix_ellpack_single_t), intent(inout) :: a
    real, optional, intent(in) :: threshold

    integer :: i, j

    a%max_bandwidth = get_max_bandwidth(a_dense)
    if(a%max_bandwidth > ELLPACK_M) then
       call bml_error(__FILE__, __LINE__, "exceeding maximum bandwidth")
    end if

    do i = 1, a%n
       do j = 1, a%n
          if(a_dense(i, j) > threshold) then
             associate(nnon0 => a%number_entries(i))
               nnon0 = nnon0+1
               a%column_index(i, nnon0) = j
               a%matrix(i, nnon0) = a_dense(i, j)
             end associate
          end if
       end do
    end do

  end subroutine convert_from_dense_ellpack_single

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \param a_dense The dense matrix.
  !! \param a The bml matrix.
  !! \param threshold The matrix element magnited threshold
  subroutine convert_from_dense_ellpack_double(a_dense, a, threshold)

    use bml_type_ellpack_m
    use bml_error_m

    double precision, intent(in) :: a_dense(:, :)
    type(bml_matrix_ellpack_double_t), intent(inout) :: a
    double precision, optional, intent(in) :: threshold

    integer :: i, j

    a%max_bandwidth = get_max_bandwidth(a_dense)
    if(a%max_bandwidth > ELLPACK_M) then
       call bml_error(__FILE__, __LINE__, "exceeding maximum bandwidth")
    end if

    do i = 1, a%n
       do j = 1, a%n
          if(a_dense(i, j) > threshold) then
             associate(nnon0 => a%number_entries(i))
               nnon0 = nnon0+1
               a%column_index(i, nnon0) = j
               a%matrix(i, nnon0) = a_dense(i, j)
             end associate
          end if
       end do
    end do

  end subroutine convert_from_dense_ellpack_double

end module bml_convert_ellpack_m
