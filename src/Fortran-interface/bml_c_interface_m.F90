!> Contains all Fortran interface defintions to the C-API of the BML-library.
!!
!! \note This module also exports the content of the iso_c_binding intrinsic
!! module, so for all other modules in the Fortran interface, it should be
!! enough to import this module when interacting with the C-API.
!!
module bml_c_interface_m

  use, intrinsic :: iso_c_binding

  implicit none

  ! Note: According to Sec. 15.3.7.2.6: "any dummy argument without
  ! the value attribute corresponds to a formal parameter of the
  ! prototype that is of a pointer type, and the dummy argument is
  ! interoperable with an entity of the referenced type (ISO/IEC
  ! 9899:1999, 6.2.5, 7.17, and 7.18.1) of the formal parameter, ..."
  !
  ! In other words, a type(C_PTR) dummy argument is interoperable with
  ! the void** type.

  interface

     subroutine bml_add_C(a, b, alpha, beta, threshold) bind(C, name="bml_add")
      import :: C_PTR, C_DOUBLE
      type(C_PTR), value, intent(in) :: a
      type(C_PTR), value, intent(in) :: b
      real(C_DOUBLE), value, intent(in) :: alpha
      real(C_DOUBLE), value, intent(in) :: beta
      real(C_DOUBLE), value, intent(in) :: threshold
    end subroutine bml_add_C

    function bml_add_norm_C(a, b, alpha, beta, threshold) &
        & bind(C, name="bml_add_norm")
      import :: C_PTR, C_DOUBLE
      type(C_PTR), value, intent(in) :: a
      type(C_PTR), value, intent(in) :: b
      real(C_DOUBLE), value, intent(in) :: alpha
      real(C_DOUBLE), value, intent(in) :: beta
      real(C_DOUBLE), value, intent(in) :: threshold
      real(C_DOUBLE) :: bml_add_norm_C
    end function bml_add_norm_C

    subroutine bml_add_identity_C(a, beta, threshold) &
        & bind(C, name="bml_add_identity")
      import :: C_PTR, C_DOUBLE
      type(C_PTR), value :: a
      real(C_DOUBLE), value, intent(in) :: beta
      real(C_DOUBLE), value, intent(in) :: threshold
    end subroutine bml_add_identity_C

    subroutine bml_scale_add_identity_C(a, alpha, beta, threshold) &
        & bind(C, name="bml_scale_add_identity")
      import :: C_PTR, C_DOUBLE
      type(C_PTR), value :: a
      real(C_DOUBLE), value, intent(in) :: alpha
      real(C_DOUBLE), value, intent(in) :: beta
      real(C_DOUBLE), value, intent(in) :: threshold
    end subroutine bml_scale_add_identity_C


    subroutine bml_adjungate_triangle_C(a, triangle) &
        & bind(C, name="bml_adjungate_triangle")
      import :: C_PTR, C_CHAR
      type(C_PTR), value :: a
      character(C_CHAR), value, intent(in) :: triangle
    end subroutine bml_adjungate_triangle_C

    subroutine bml_transpose_triangle_C(a, triangle) &
        & bind(C, name="bml_transpose_triangle")
      import :: C_PTR, C_CHAR
      type(C_PTR), value :: a
      character(C_CHAR), value, intent(in) :: triangle
    end subroutine bml_transpose_triangle_C

    function bml_banded_matrix_C(matrix_type, matrix_precision, n, m) &
        & bind(C, name="bml_banded_matrix")
      import :: C_INT, C_PTR
      integer(C_INT), value, intent(in) :: matrix_type
      integer(C_INT), value, intent(in) :: matrix_precision
      integer(C_INT), value, intent(in) :: n
      integer(C_INT), value, intent(in) :: m
      type(C_PTR) :: bml_banded_matrix_C
    end function bml_banded_matrix_C

    function bml_convert_from_dense_C(matrix_type, matrix_precision, order, &
        & n, a, threshold, m) bind(C, name="bml_convert_from_dense")
      import :: C_INT, C_PTR, C_DOUBLE
      integer(C_INT), value, intent(in) :: matrix_type
      integer(C_INT), value, intent(in) :: matrix_precision
      integer(C_INT), value, intent(in) :: order
      integer(C_INT), value, intent(in) :: n, m
      type(C_PTR), value, intent(in) :: a
      real(C_DOUBLE), value, intent(in) :: threshold
      type(C_PTR) :: bml_convert_from_dense_C
    end function bml_convert_from_dense_C

    function bml_convert_to_dense_C(a, order) &
        & bind(C, name="bml_convert_to_dense")
      import :: C_INT, C_PTR
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: order
      type(C_PTR) :: bml_convert_to_dense_C
    end function bml_convert_to_dense_C

    subroutine bml_copy_C(a, b) bind(C, name="bml_copy")
      import :: C_PTR
      type(C_PTR), value, intent(in) :: a
      type(C_PTR), value, intent(in) :: b
    end subroutine bml_copy_C

    function bml_copy_new_C(a) bind(C, name="bml_copy_new")
      import :: C_PTR
      type(C_PTR), value, intent(in) :: a
      type(C_PTR) :: bml_copy_new_C
    end function bml_copy_new_C

    subroutine bml_diagonalize_C(a, eigenvalues, eigenvectors) &
        & bind(C, name="bml_diagonalize")
      import :: C_PTR
      type(C_PTR), value, intent(in) :: a
      type(C_PTR), value :: eigenvalues
      type(C_PTR), value :: eigenvectors
    end subroutine bml_diagonalize_C

    subroutine bml_deallocate_C(a) bind(C, name="bml_deallocate")
      import :: C_PTR
      type(C_PTR) :: a
    end subroutine bml_deallocate_C

    function bml_get_single_real_C(a, i, j) bind(C, name="bml_get_single_real")
      import :: C_PTR, C_INT, C_FLOAT
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: i
      integer(C_INT), value, intent(in) :: j
      real(C_FLOAT) :: bml_get_single_real_C
    end function bml_get_single_real_C

    function bml_get_double_real_C(a, i, j) bind(C, name="bml_get_double_real")
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: i
      integer(C_INT), value, intent(in) :: j
      real(C_DOUBLE) :: bml_get_double_real_C
    end function bml_get_double_real_C

    function bml_get_single_complex_C(a, i, j) &
        & bind(C, name="bml_get_single_complex")
      import :: C_PTR, C_INT, C_FLOAT_COMPLEX
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: i
      integer(C_INT), value, intent(in) :: j
      complex(C_FLOAT_COMPLEX) :: bml_get_single_complex_C
    end function bml_get_single_complex_C

    function bml_get_double_complex_C(a, i, j) &
        & bind(C, name="bml_get_double_complex")
      import :: C_PTR, C_INT, C_DOUBLE_COMPLEX
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: i
      integer(C_INT), value, intent(in) :: j
      complex(C_DOUBLE_COMPLEX) :: bml_get_double_complex_C
    end function bml_get_double_complex_C

    function bml_get_N_C(a) bind(C, name="bml_get_N")
      import :: C_PTR, C_INT
      type(C_PTR), value, intent(in) :: a
      integer(C_INT) :: bml_get_N_C
    end function bml_get_N_C

    subroutine bml_get_row_C(a, i, row) bind(C, name="bml_get_row")
      import :: C_PTR, C_INT
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: i
      type(C_PTR), value, intent(in) :: row
    end subroutine bml_get_row_C

     subroutine bml_normalize_C(a, maxeval, maxminusmin) &
       bind(C, name="bml_normalize")
       import :: C_PTR, C_DOUBLE
       type(C_PTR), value :: a
       real(C_DOUBLE), value, intent(in) :: maxeval
       real(C_DOUBLE), value, intent(in) :: maxminusmin
     end subroutine bml_normalize_C

    function bml_gershgorin_C(a) bind(C, name="bml_gershgorin")
      import :: C_PTR
      type(C_PTR), value, intent(in) :: a
      type(C_PTR) :: bml_gershgorin_C
    end function bml_gershgorin_C

    function bml_get_row_bandwidth_C(a, i) &
        & bind(C, name="bml_get_row_bandwidth")
      import :: C_PTR, C_INT
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: i
      integer(C_INT) :: bml_get_row_bandwidth_C
    end function bml_get_row_bandwidth_C

    function bml_get_bandwidth_C(a) bind(C, name="bml_get_bandwidth")
      import :: C_PTR, C_INT
      type(C_PTR), value, intent(in) :: a
      integer(C_INT) :: bml_get_bandwidth_C
    end function bml_get_bandwidth_C

    function bml_identity_matrix_C(matrix_type, matrix_precision, n, m) &
        & bind(C, name="bml_identity_matrix")
      import :: C_INT, C_PTR
      integer(C_INT), value, intent(in) :: matrix_type
      integer(C_INT), value, intent(in) :: matrix_precision
      integer(C_INT), value, intent(in) :: n
      integer(C_INT), value, intent(in) :: m
      type(C_PTR) :: bml_identity_matrix_C
    end function bml_identity_matrix_C

    subroutine bml_multiply_C(a, b, c, alpha, beta, threshold) &
        & bind(C, name="bml_multiply")
      import :: C_PTR, C_DOUBLE
      type(C_PTR), value, intent(in) :: a
      type(C_PTR), value, intent(in) :: b
      type(C_PTR), value, intent(in) :: c
      real(C_DOUBLE), value, intent(in) :: alpha
      real(C_DOUBLE), value, intent(in) :: beta
      real(C_DOUBLE), value, intent(in) :: threshold
    end subroutine bml_multiply_C

    function bml_multiply_x2_C(x, x2, threshold) &
        & bind(C, name="bml_multiply_x2")
      import :: C_PTR, C_DOUBLE
      type(C_PTR), value, intent(in) :: x
      type(C_PTR), value, intent(in) :: x2
      real(C_DOUBLE), value, intent(in) :: threshold
      type(C_PTR) :: bml_multiply_x2_C
    end function bml_multiply_x2_C

    function bml_random_matrix_C(matrix_type, matrix_precision, n, m) &
        & bind(C, name="bml_random_matrix")
      import :: C_INT, C_PTR
      integer(C_INT), value, intent(in) :: matrix_type
      integer(C_INT), value, intent(in) :: matrix_precision
      integer(C_INT), value, intent(in) :: n
      integer(C_INT), value, intent(in) :: m
      type(C_PTR) :: bml_random_matrix_C
    end function bml_random_matrix_C

    subroutine bml_scale_C(alpha, a, b) bind(C, name="bml_scale")
      import :: C_PTR, C_DOUBLE
      real(C_DOUBLE), value, intent(in) :: alpha
      type(C_PTR), value :: a
      type(C_PTR), value :: b
    end subroutine bml_scale_C

    subroutine bml_scale_inplace_C(alpha, a) bind(C, name="bml_scale_inplace")
      import :: C_PTR, C_DOUBLE
      real(C_DOUBLE), value, intent(in) :: alpha
      type(C_PTR), value :: a
    end subroutine bml_scale_inplace_C

    subroutine bml_set_row_C(a, i, row, threshold) bind(C, name="bml_set_row")
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: i
      type(C_PTR), value :: row
      real(C_DOUBLE), value :: threshold
    end subroutine bml_set_row_C

    subroutine bml_set_diagonal_C(a, diagonal, threshold) bind(C, name="bml_set_diagonal")
      import :: C_PTR, C_INT, C_DOUBLE
      type(C_PTR), value, intent(in) :: a
      type(C_PTR), value :: diagonal
      real(C_DOUBLE), value :: threshold
    end subroutine bml_set_diagonal_C

    subroutine bml_threshold_C(a, threshold) bind(C, name="bml_threshold")
      import :: C_PTR, C_DOUBLE
      type(C_PTR), value :: a
      real(C_DOUBLE), value, intent(in) :: threshold
    end subroutine bml_threshold_C

    subroutine bml_print_bml_vector_C(v, i_l, i_u) &
        & bind(C, name="bml_print_bml_vector")
      import :: C_PTR, C_INT
      type(C_PTR), value, intent(in) :: v
      integer(C_INT), value, intent(in) :: i_l
      integer(C_INT), value, intent(in) :: i_u
    end subroutine bml_print_bml_vector_C

    subroutine bml_print_bml_matrix_C(a, i_l, i_u, j_l, j_u) &
        & bind(C, name="bml_print_bml_matrix")
      import :: C_PTR, C_INT
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: i_l
      integer(C_INT), value, intent(in) :: i_u
      integer(C_INT), value, intent(in) :: j_l
      integer(C_INT), value, intent(in) :: j_u
    end subroutine bml_print_bml_matrix_C

    subroutine bml_print_dense_matrix_C(n, matrix_precision, order, a, i_l, &
        & i_u, j_l, j_u) bind(C, name="bml_print_dense_matrix")
      import :: C_PTR, C_INT
      integer(C_INT), value, intent(in) :: n
      integer(C_INT), value, intent(in) :: matrix_precision
      integer(C_INT), value, intent(in) :: order
      type(C_PTR), value, intent(in) :: a
      integer(C_INT), value, intent(in) :: i_l
      integer(C_INT), value, intent(in) :: i_u
      integer(C_INT), value, intent(in) :: j_l
      integer(C_INT), value, intent(in) :: j_u
    end subroutine bml_print_dense_matrix_C

    subroutine bml_read_bml_matrix_C(a, filename) &
        & bind(C, name="bml_read_bml_matrix")
      import :: C_PTR, C_CHAR
      type(C_PTR), value, intent(in) :: a
      character(C_CHAR), intent(in) :: filename(*)
    end subroutine bml_read_bml_matrix_C

    function bml_trace_C(a) bind(C, name="bml_trace")
      import :: C_PTR, C_DOUBLE
      type(C_PTR), value, intent(in) :: a
      real(C_DOUBLE) :: bml_trace_C
    end function bml_trace_C

    function bml_transpose_new_C(a) bind(C, name="bml_transpose_new")
      import :: C_PTR
      type(C_PTR), value, intent(in) :: a
      type(C_PTR) :: bml_transpose_new_C
    end function bml_transpose_new_C

    subroutine bml_write_bml_matrix_C(a, filename) &
        & bind(C, name="bml_write_bml_matrix")
      import :: C_PTR, C_CHAR
      type(C_PTR), value, intent(in) :: a
      character(C_CHAR), intent(in) :: filename(*)
    end subroutine bml_write_bml_matrix_C

    function bml_zero_matrix_C(matrix_type, matrix_precision, n, m) &
        & bind(C, name="bml_zero_matrix")
      import :: C_INT, C_PTR
      integer(C_INT), value, intent(in) :: matrix_type
      integer(C_INT), value, intent(in) :: matrix_precision
      integer(C_INT), value, intent(in) :: n
      integer(C_INT), value, intent(in) :: m
      type(C_PTR) :: bml_zero_matrix_C
    end function bml_zero_matrix_C

  end interface

end module bml_c_interface_m
