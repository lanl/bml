module bml_convert_m
  use bml_c_interface_m
  use bml_types_m
  use bml_interface_m
  use bml_introspection_m
  implicit none
  private

  interface bml_convert_from_dense
     module procedure bml_convert_from_dense_single
     module procedure bml_convert_from_dense_double
     module procedure bml_convert_from_dense_single_complex
     module procedure bml_convert_from_dense_double_complex
  end interface bml_convert_from_dense

  interface bml_convert_to_dense
     module procedure bml_convert_to_dense_single
     module procedure bml_convert_to_dense_double
     module procedure bml_convert_to_dense_single_complex
     module procedure bml_convert_to_dense_double_complex
  end interface bml_convert_to_dense

  public :: bml_convert_from_dense
  public :: bml_convert_to_dense

contains

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param matrix_type The matrix type
  !! \param a_dense The dense matrix
  !! \param a The bml matrix
  !! \param threshold The matrix element magnited threshold
  !! \param m The extra arg
  subroutine bml_convert_from_dense_single(matrix_type, a_dense, a, threshold, m)
    character(len=*), intent(in) :: matrix_type
    real(C_FLOAT), target, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    real(C_DOUBLE), optional, intent(in) :: threshold
    integer, optional, intent(in) :: m

    integer(C_INT) :: m_
    real(C_DOUBLE) :: threshold_

    if(present(threshold)) then
       threshold_ = threshold
    else
       threshold_ = 0
    end if

    if(present(m)) then
       m_ = m
    else
       m_ = 0
    end if

    associate(a_ptr => a_dense(lbound(a_dense, 1), lbound(a_dense, 2)))
      a%ptr = bml_convert_from_dense_C(get_matrix_id(matrix_type), &
          & get_element_id(BML_ELEMENT_REAL, C_FLOAT), BML_DENSE_COLUMN_MAJOR, &
          & size(a_dense, 1, C_INT), c_loc(a_ptr), threshold_, m_)
    end associate

  end subroutine bml_convert_from_dense_single

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param matrix_type The matrix type
  !! \param a_dense The dense matrix
  !! \param a The bml matrix
  !! \param threshold The matrix element magnited threshold
  !! \param m the extra arg
  subroutine bml_convert_from_dense_double(matrix_type, a_dense, a, threshold, m)
    character(len=*), intent(in) :: matrix_type
    real(C_DOUBLE), target, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    real(C_DOUBLE), optional, intent(in) :: threshold
    integer, optional, intent(in) :: m

    integer(C_INT) :: m_
    real(C_DOUBLE) :: threshold_

    if(present(threshold)) then
       threshold_ = threshold
    else
       threshold_ = 0
    end if

    if(present(m)) then
       m_ = m
    else
       m_ = 0
    end if

    associate(a_ptr => a_dense(lbound(a_dense, 1), lbound(a_dense, 2)))
      a%ptr = bml_convert_from_dense_C(get_matrix_id(matrix_type), &
          & get_element_id(BML_ELEMENT_REAL, C_DOUBLE), &
          & BML_DENSE_COLUMN_MAJOR, size(a_dense, 1, kind=C_INT), &
          & c_loc(a_ptr), threshold_, m_)
    end associate

  end subroutine bml_convert_from_dense_double

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param matrix_type The matrix type
  !! \param a_dense The dense matrix
  !! \param a The bml matrix
  !! \param threshold The matrix element magnited threshold
  !! \param m The extra arg
  subroutine bml_convert_from_dense_single_complex(matrix_type, a_dense, a, threshold, m)

    character(len=*), intent(in) :: matrix_type
    complex(C_FLOAT_COMPLEX), target, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    real(C_DOUBLE), optional, intent(in) :: threshold
    integer, optional, intent(in) :: m

    integer(C_INT) :: m_
    real(C_DOUBLE) :: threshold_

    if(present(threshold)) then
       threshold_ = threshold
    else
       threshold_ = 0
    end if

    if(present(m)) then
       m_ = m
    else
       m_ = 0
    end if

    associate(a_ptr => a_dense(lbound(a_dense, 1), lbound(a_dense, 2)))
      a%ptr = bml_convert_from_dense_C(get_matrix_id(matrix_type), &
          & get_element_id(BML_ELEMENT_COMPLEX, C_FLOAT_COMPLEX), &
          & BML_DENSE_COLUMN_MAJOR, size(a_dense, 1, kind=C_INT), &
          & c_loc(a_ptr), threshold_, m_)
    end associate

  end subroutine bml_convert_from_dense_single_complex

  !> Convert a dense matrix into a bml matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param matrix_type The matrix type
  !! \param a_dense The dense matrix
  !! \param a The bml matrix
  !! \param threshold The matrix element magnited threshold
  !! \param m the extra arg
  subroutine bml_convert_from_dense_double_complex(matrix_type, a_dense, a, threshold, m)

    character(len=*), intent(in) :: matrix_type
    complex(C_DOUBLE_COMPLEX), target, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    real(C_DOUBLE), optional, intent(in) :: threshold
    integer, optional, intent(in) :: m

    integer(C_INT) :: m_
    real(C_DOUBLE) :: threshold_

    if(present(threshold)) then
       threshold_ = threshold
    else
       threshold_ = 0
    end if

    if(present(m)) then
       m_ = m
    else
       m_ = 0
    end if

    associate(a_ptr => a_dense(lbound(a_dense, 1), lbound(a_dense, 2)))
      a%ptr = bml_convert_from_dense_C(get_matrix_id(matrix_type), &
          & get_element_id(BML_ELEMENT_COMPLEX, C_DOUBLE_COMPLEX), &
          & BML_DENSE_COLUMN_MAJOR, size(a_dense, 1, kind=C_INT), &
          & c_loc(a_ptr), threshold_, m_)
    end associate

  end subroutine bml_convert_from_dense_double_complex

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_single(a, a_dense)

    type(bml_matrix_t), intent(in) :: a
    real(C_FLOAT), allocatable, intent(inout) :: a_dense(:, :)

    type(C_PTR) :: a_ptr
    real(C_FLOAT), pointer :: a_dense_ptr(:, :)

    a_ptr = bml_convert_to_dense_C(a%ptr, BML_DENSE_COLUMN_MAJOR)
    call c_f_pointer(a_ptr, a_dense_ptr, [bml_get_n(a), bml_get_n(a)])
    a_dense = a_dense_ptr

  end subroutine bml_convert_to_dense_single

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_double(a, a_dense)

    type(bml_matrix_t), intent(in) :: a
    real(C_DOUBLE), allocatable, intent(inout) :: a_dense(:, :)

    type(C_PTR) :: a_ptr
    real(C_DOUBLE), pointer :: a_dense_ptr(:, :)

    a_ptr = bml_convert_to_dense_C(a%ptr, BML_DENSE_COLUMN_MAJOR)
    call c_f_pointer(a_ptr, a_dense_ptr, [bml_get_n(a), bml_get_n(a)])
    a_dense = a_dense_ptr

  end subroutine bml_convert_to_dense_double

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_single_complex(a, a_dense)

    type(bml_matrix_t), intent(in) :: a
    complex(C_FLOAT_COMPLEX), allocatable, intent(out) :: a_dense(:, :)

    type(C_PTR) :: a_ptr
    complex(C_FLOAT_COMPLEX), pointer :: a_dense_ptr(:, :)

    a_ptr = bml_convert_to_dense_C(a%ptr, BML_DENSE_COLUMN_MAJOR)
    call c_f_pointer(a_ptr, a_dense_ptr, [bml_get_n(a), bml_get_n(a)])
    a_dense = a_dense_ptr

  end subroutine bml_convert_to_dense_single_complex

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_double_complex(a, a_dense)

    type(bml_matrix_t), intent(in) :: a
    complex(C_DOUBLE_COMPLEX), allocatable, intent(out) :: a_dense(:, :)

    type(C_PTR) :: a_ptr
    complex(C_DOUBLE_COMPLEX), pointer :: a_dense_ptr(:, :)

    a_ptr = bml_convert_to_dense_C(a%ptr, BML_DENSE_COLUMN_MAJOR)
    call c_f_pointer(a_ptr, a_dense_ptr, [bml_get_n(a), bml_get_n(a)])
    a_dense = a_dense_ptr

  end subroutine bml_convert_to_dense_double_complex

end module bml_convert_m
