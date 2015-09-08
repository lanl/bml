!> \copyright Los Alamos National Laboratory 2015

!> Getter functions for matrix elements.
module bml_get_bandwidth_m

  implicit none

  private

  public :: bml_get_bandwidth

contains

  !> Get the number of non-zero elements in a given row.
  !!
  !! @param a The matrix.
  !! @param i The row.
  !! @returns The number of non-zero elements (bandwidth) on that row.
  function bml_get_bandwidth(a, i) result(m_i)

    use bml_type_m
    use bml_type_dense_m
    use bml_type_ellpack_m
    use bml_get_bandwidth_dense_m
    use bml_get_bandwidth_ellpack_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    integer :: m_i

    select type(a)
    class is(bml_matrix_dense_t)
       m_i = bml_get_bandwidth_dense(a, i)
    class is(bml_matrix_ellpack_t)
       m_i = bml_get_bandwidth_ellpack(a, i)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end function bml_get_bandwidth

end module bml_get_bandwidth_m
