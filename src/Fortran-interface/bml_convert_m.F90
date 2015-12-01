module bml_convert_m

  implicit none

  private

  !> The interfaces to the C API.
  interface

     function bml_convert_from_dense_C(matrix_type, matrix_precision, order, n, a, threshold, m) &
          bind(C, name="bml_convert_from_dense")
       use, intrinsic :: iso_C_binding
       integer(C_INT), value, intent(in) :: matrix_type
       integer(C_INT), value, intent(in) :: matrix_precision
       integer(C_INT), value, intent(in) :: order
       integer(C_INT), value, intent(in) :: n, m
       type(C_PTR), value, intent(in) :: a
       real(C_DOUBLE), value, intent(in) :: threshold
       type(C_PTR) :: bml_convert_from_dense_C
     end function bml_convert_from_dense_C

     function bml_convert_to_dense_C(a, order) bind(C, name="bml_convert_to_dense")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: order
       type(C_PTR) :: bml_convert_to_dense_C
     end function bml_convert_to_dense_C

  end interface

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

    use bml_types_m
    use bml_interface_m

    character(len=*), intent(in) :: matrix_type
    real, target, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    double precision, optional, intent(in) :: threshold
    integer, optional, intent(in) :: m

    integer :: m_
    double precision :: threshold_

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
      a%ptr = bml_convert_from_dense_C(get_enum_id(matrix_type), &
           get_enum_id(BML_PRECISION_SINGLE_REAL), BML_DENSE_COLUMN_MAJOR, &
           size(a_dense, 1), c_loc(a_ptr), threshold_, m_)
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

    use bml_types_m
    use bml_interface_m

    character(len=*), intent(in) :: matrix_type
    double precision, target, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    double precision, optional, intent(in) :: threshold
    integer, optional, intent(in) :: m

    integer :: m_
    double precision :: threshold_

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
      a%ptr = bml_convert_from_dense_C(get_enum_id(matrix_type), &
           get_enum_id(BML_PRECISION_DOUBLE_REAL), BML_DENSE_COLUMN_MAJOR, &
           size(a_dense, 1), c_loc(a_ptr), threshold_, m_)
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

    use bml_types_m
    use bml_interface_m

    character(len=*), intent(in) :: matrix_type
    complex, target, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    double precision, optional, intent(in) :: threshold
    integer, optional, intent(in) :: m

    integer :: m_
    double precision :: threshold_

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
      a%ptr = bml_convert_from_dense_C(get_enum_id(matrix_type), &
           get_enum_id(BML_PRECISION_SINGLE_COMPLEX), BML_DENSE_COLUMN_MAJOR, &
           size(a_dense, 1), c_loc(a_ptr), threshold_, m_)
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

    use bml_types_m
    use bml_interface_m

    character(len=*), intent(in) :: matrix_type
    complex(kind(0.0d0)), target, intent(in) :: a_dense(:, :)
    type(bml_matrix_t), intent(inout) :: a
    double precision, optional, intent(in) :: threshold
    integer, optional, intent(in) :: m

    integer :: m_
    double precision :: threshold_

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
      a%ptr = bml_convert_from_dense_C(get_enum_id(matrix_type), &
           get_enum_id(BML_PRECISION_DOUBLE_COMPLEX), BML_DENSE_COLUMN_MAJOR, &
           size(a_dense, 1), c_loc(a_ptr), threshold_, m_)
    end associate

  end subroutine bml_convert_from_dense_double_complex

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_single(a, a_dense)

    use bml_types_m
    use bml_interface_m
    use bml_introspection_m

    type(bml_matrix_t), intent(in) :: a
    real, allocatable, intent(inout) :: a_dense(:, :)

    type(C_PTR) :: a_ptr
    real, pointer :: a_dense_ptr(:, :)

    a_ptr = bml_convert_to_dense_C(a%ptr, BML_DENSE_COLUMN_MAJOR)
    call c_f_pointer(a_ptr, a_dense_ptr, [bml_get_n(a), bml_get_n(a)])
    a_dense = a_dense_ptr
    deallocate(a_dense_ptr)

  end subroutine bml_convert_to_dense_single

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_double(a, a_dense)

    use bml_types_m
    use bml_interface_m
    use bml_introspection_m

    type(bml_matrix_t), intent(in) :: a
    double precision, allocatable, intent(inout) :: a_dense(:, :)

    type(C_PTR) :: a_ptr
    double precision, pointer :: a_dense_ptr(:, :)

    a_ptr = bml_convert_to_dense_C(a%ptr, BML_DENSE_COLUMN_MAJOR)
    call c_f_pointer(a_ptr, a_dense_ptr, [bml_get_n(a), bml_get_n(a)])
    a_dense = a_dense_ptr
    deallocate(a_dense_ptr)

  end subroutine bml_convert_to_dense_double

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_single_complex(a, a_dense)

    use bml_types_m
    use bml_interface_m
    use bml_introspection_m

    type(bml_matrix_t), intent(in) :: a
    complex, allocatable, intent(out) :: a_dense(:, :)

    type(C_PTR) :: a_ptr
    complex, pointer :: a_dense_ptr(:, :)

    a_ptr = bml_convert_to_dense_C(a%ptr, BML_DENSE_COLUMN_MAJOR)
    call c_f_pointer(a_ptr, a_dense_ptr, [bml_get_n(a), bml_get_n(a)])
    a_dense = a_dense_ptr
    deallocate(a_dense_ptr)

  end subroutine bml_convert_to_dense_single_complex

  !> Convert a matrix into a dense matrix.
  !!
  !! \ingroup convert_group_Fortran
  !!
  !! \param a The bml matrix
  !! \param a_dense The dense matrix
  subroutine bml_convert_to_dense_double_complex(a, a_dense)

    use bml_types_m
    use bml_interface_m
    use bml_introspection_m

    type(bml_matrix_t), intent(in) :: a
    complex(kind(0d0)), allocatable, intent(out) :: a_dense(:, :)

    type(C_PTR) :: a_ptr
    complex(kind(0d0)), pointer :: a_dense_ptr(:, :)

    a_ptr = bml_convert_to_dense_C(a%ptr, BML_DENSE_COLUMN_MAJOR)
    call c_f_pointer(a_ptr, a_dense_ptr, [bml_get_n(a), bml_get_n(a)])
    a_dense = a_dense_ptr
    deallocate(a_dense_ptr)

  end subroutine bml_convert_to_dense_double_complex

end module bml_convert_m
