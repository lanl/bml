program main

  use bml

  implicit none
  integer, parameter :: dp = kind(1.0d0)
  type(bml_matrix_t) :: hmat_bml, aux_bml, aux1_bml, aux2_bml
  complex(dp), allocatable :: H(:,:), aux(:,:), diag(:)
  complex(dp) :: cf
  real(dp) :: rval
  character(20) :: bml_type
  real(dp) :: numthresh
  integer :: i,j,hdim,mdim


  write(*,*)''
  write(*,*)'Example Fortran BML 4: Read a dense matrix form file, convert to bml_dense'
  write(*,*)'; extract the diagonal'
  write(*,*)'This can be also done for dense by changing the ellpack type'
  write(*,*)''

  open(1,file='hamiltonian.dat')
  read(1,*)hdim
  mdim = hdim
  bml_type = "ellpack"
  numthresh = 0.000001_dp

  allocate(H(hdim,hdim))
  allocate(diag(hdim))

  H=0.0_dp

  do i=1,hdim
     do j=1,hdim
        read(1,*)rval
        H(i,j) = cmplx(rval,0.0_dp)
     enddo
  enddo

  call bml_zero_matrix(bml_type,bml_element_complex,dp,hdim,mdim,hmat_bml)

  call bml_import_from_dense(bml_type,H,hmat_bml,numthresh,mdim)

  call bml_get_diagonal(hmat_bml,diag)

  write(*,*)"diagonal               = (",h(1,1),h(2,2),h(3,3),"...)"
  write(*,*)"diagonal_bml_extracted = (",diag(1),diag(2),diag(3),"...)"

end program main
