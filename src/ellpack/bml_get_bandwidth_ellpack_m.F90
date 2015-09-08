module bml_get_bandwidth_ellpack_m

  implicit none

  private

  public :: bml_get_bandwidth_ellpack

contains

  !> Get the number of non-zero elements in a given row.
  !!
  !! @param a The matrix.
  !! @param i The row.
  !! @returns The number of non-zero elements (bandwidth) on that row.
  function bml_get_bandwidth_ellpack(a, i) result(m_i)

    use bml_type_ellpack_m

    class(bml_matrix_ellpack_t), intent(in) :: a
    integer, intent(in) :: i
    integer :: m_i

    m_i = a%number_entries(i)

  end function bml_get_bandwidth_ellpack

end module bml_get_bandwidth_ellpack_m
