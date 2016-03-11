!> Introspection procedures.
module bml_introspection_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_get_N
  public :: bml_get_type
  public :: bml_get_precision
  public :: bml_get_row_bandwidth
  public :: bml_get_bandwidth

contains

  !> Return the matrix size.
  !!
  !!\param a The matrix.
  !!\return The matrix size.
  function bml_get_n(a)

    type(bml_matrix_t), intent(in) :: a
    integer :: bml_get_n

    bml_get_n = bml_get_N_C(a%ptr)

  end function bml_get_n

  !> Get the bandwidth of non-zero elements in a given row.
  !!
  !! @param a The matrix.
  !! @param i The row.
  !! @returns The bandwidth of non-zero elements (bandwidth) on that row.
  function bml_get_row_bandwidth(a, i)

    type(bml_matrix_t), intent(in) :: a
    integer(C_INT), intent(in) :: i
    integer :: bml_get_row_bandwidth

    bml_get_row_bandwidth = bml_get_row_bandwidth_C(a%ptr, i-1)

  end function bml_get_row_bandwidth

  !> Get the bandwidth of non-zero elements of a matrix.
  !!
  !! @param a The matrix.
  !! @returns The bandwidth of non-zero elements (bandwidth) of the matrix.
  function bml_get_bandwidth(a)

    type(bml_matrix_t), intent(in) :: a
    integer :: bml_get_bandwidth

    bml_get_bandwidth = bml_get_bandwidth_C(a%ptr)

  end function bml_get_bandwidth

  !> Get the type of a matrix.
  !!
  !! @param a The matrix.
  !! @returns The bml format type of the matrix.
  function bml_get_type(a)

    type(bml_matrix_t), intent(in) :: a
    integer :: bml_get_type_num
    character(20) :: bml_get_type

    bml_get_type_num = bml_get_type_C(a%ptr)

    select case(bml_get_type_num)
      case(0)
        bml_get_type = "Unformatted"
      case(1)
        bml_get_type = "dense"
      case(2)
        bml_get_type = "ellpack"
      case default
        stop 'Unknown matrix type in bml_get_type'
    end select

  end function bml_get_type

!> Get the precision of a matrix.
  !!
  !! @param a The matrix.
  !! @returns The bml format precision of the matrix.
  function bml_get_precision(a)

    type(bml_matrix_t), intent(in) :: a
    integer :: bml_get_precision

    bml_get_precision = bml_get_precision_C(a%ptr)

  end function bml_get_precision

end module bml_introspection_m
