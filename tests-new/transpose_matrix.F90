module transpose_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: transpose_matrix_t
   contains
     procedure, nopass :: test_function
  end type transpose_matrix_t

contains

  function test_function(n, matrix_type, matrix_precision, m) result(test_result)

    integer, intent(in) :: n, m
    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    logical :: test_result

    class(bml_matrix_t), allocatable :: a
    class(bml_matrix_t), allocatable :: b

    real, allocatable :: a_real(:, :)
    real, allocatable :: b_real(:, :)
    double precision, allocatable :: a_double(:, :)
    double precision, allocatable :: b_double(:, :)

    call bml_random_matrix(matrix_type, n, a, matrix_precision, m)
    call bml_transpose(a, b)

    select case(matrix_precision)
    case(BML_PRECISION_SINGLE)
       call bml_convert_to_dense(a, a_real)
       call bml_convert_to_dense(b, b_real)
       if(maxval(a_real-transpose(b_real)) > 1e-12) then
          test_result = .false.
          print *, "matrices are not transposes"
       else
          test_result = .true.
       end if
    case(BML_PRECISION_DOUBLE)
       call bml_convert_to_dense(a, a_double)
       call bml_convert_to_dense(b, b_double)
       if(maxval(a_double-transpose(b_double)) > 1e-12) then
          test_result = .false.
          print *, "matrices are not transposes"
       else
          test_result = .true.
       end if
    end select

  end function test_function

end module transpose_matrix_m
