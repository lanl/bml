!> \copyright Los Alamos National Laboratory 2015

!> Getter functions for matrix elements.
module bml_get_bandwidth_dense_m

  implicit none

  private

  public :: bml_get_bandwidth_dense

contains

  !> Get the number of non-zero elements in a given row.
  !!
  !! @param a The matrix.
  !! @param i The row.
  !! @returns The number of non-zero elements (bandwidth) on that row.
  function bml_get_bandwidth_dense(a, i) result(m_i)

    use bml_type_dense_m

    class(bml_matrix_dense_t), intent(in) :: a
    integer, intent(in) :: i
    integer :: m_i

    select type(a)
    type is(bml_matrix_dense_single_t)
       m_i = get_bandwidth_dense_single(a, i)
    type is(bml_matrix_dense_double_t)
       m_i = get_bandwidth_dense_double(a, i)
    end select

  end function bml_get_bandwidth_dense

  !> Get the number of non-zero elements in a given row.
  !!
  !! @param a The matrix.
  !! @param i The row.
  !! @returns The number of non-zero elements (bandwidth) on that row.
  function get_bandwidth_dense_single(a, i) result(m_i)

    use bml_type_dense_m

    type(bml_matrix_dense_single_t), intent(in) :: a
    integer, intent(in) :: i
    integer :: m_i

    integer :: j

    m_i = 0
    do j = 1, a%n
       if(a%matrix(i, j) /= 0) then
          m_i = m_i+1
       end if
    end do

  end function get_bandwidth_dense_single

  !> Get the number of non-zero elements in a given row.
  !!
  !! @param a The matrix.
  !! @param i The row.
  !! @returns The number of non-zero elements (bandwidth) on that row.
  function get_bandwidth_dense_double(a, i) result(m_i)

    use bml_type_dense_m

    type(bml_matrix_dense_double_t), intent(in) :: a
    integer, intent(in) :: i
    integer :: m_i

    integer :: j

    m_i = 0
    do j = 1, a%n
       if(a%matrix(i, j) /= 0) then
          m_i = m_i+1
       end if
    end do

  end function get_bandwidth_dense_double

end module bml_get_bandwidth_dense_m
