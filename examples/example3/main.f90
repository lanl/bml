! Program to compare timings between bml wrapper to dgemms
! and dgemms standalone.

program main

  use bml

  implicit none
  integer, parameter :: dp = kind(1.0d0)
  character(20)          ::  bml_type
  integer                ::  hdim, i, j, mdim
  real(8)                ::  mls, mlsI
  real(dp)               ::  alpha, beta, numthresh
  real(dp), allocatable  ::  aux(:,:), diff(:,:), h(:,:), trace(:)
  real(dp), allocatable  ::  x2(:,:)
  type(bml_matrix_t)     ::  hmat_bml, x2_bml

  write(*,*)''
  write(*,*)'Example BML 4: Construct a random matrix;'
  write(*,*)'multiply the matrix by itself and compare times to dgemm'
  write(*,*)''

  hdim = 4000
  mdim = 4000
  bml_type = "dense"
  numthresh = 0.0_dp
  alpha = 1.0_dp
  beta = 0.0_dp

  allocate(h(hdim,hdim))
  allocate(aux(hdim,hdim))
  allocate(diff(hdim,hdim))
  allocate(x2(hdim,hdim))

  H=0.0_dp

  call bml_random_matrix(bml_type, bml_element_real, dp, hdim, mdim, hmat_bml)
  call bml_export_to_dense(hmat_bml,H)
  call bml_copy_new(hmat_bml,x2_bml)

  !With BML wrapper to dgemm
  mlsI = mls()
  call bml_multiply_x2(hmat_bml, x2_bml, numthresh, trace)
  write(*,*)"Time for x2 bml dense=", mls()-mlsI

  call bml_export_to_dense(x2_bml,x2)

  !With dgemm
  mlsI = mls()
  call MMult(alpha,h,h,beta,aux,'N','N',hdim)
  write(*,*)"Time for x2 dgemm=", mls()-mlsI

  !Fnorm to see the difference
  diff = aux - x2
  call bml_import_from_dense(bml_type,diff,hmat_bml,numthresh,mdim)

  write(*,*)"Fnorm difference=", bml_fnorm(hmat_bml)

end program main


subroutine MMult(alpha,a,b,beta,c,ta,tb,hdim)
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  character(1), intent(in)  ::  ta, tb
  integer, intent(in)       ::  hdim
  real(dp), intent(inout)   ::  a(hdim, hdim), alpha, b(hdim,hdim), beta
  real(dp), intent(inout)   ::  c(hdim, hdim)

  call dgemm(ta, tb, hdim, hdim, hdim, alpha, &
                a, hdim, b, hdim, beta, c, hdim)

end subroutine MMult

function mls()
  integer, parameter :: dp = kind(1.0d0)
  real(dp) :: mls
  integer :: timevector(8)

  mls = 0.0_dp
  call date_and_time(values=timevector)
  mls=timevector(5)*60.0_dp*60.0_dp*1000.0_dp + timevector(6)*60.0_dp*1000.0_dp &
         + timevector(7)*1000.0_dp + timevector(8)

end function mls
