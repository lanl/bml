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
  write(*,*)'; add a complex bml matrix'
  write(*,*)'This can be also done for dense by changing the ellpack type'
  write(*,*)''

  open(1,file='hamiltonian.dat')
  read(1,*)hdim
  mdim = hdim
  bml_type = "ellpack"
  numthresh = 0.000001_dp

  allocate(H(hdim,hdim))
  allocate(aux(hdim,hdim))
  allocate(diag(hdim))

  H=0.0_dp

  do i=1,hdim
     do j=1,hdim
        read(1,*)rval
        H(i,j) = cmplx(rval,0.0_dp)
     enddo
  enddo

  call bml_zero_matrix(bml_type,bml_element_complex,dp,hdim,mdim,hmat_bml)
  call bml_zero_matrix(bml_type,bml_element_complex,dp,hdim,mdim,aux_bml)
  call bml_zero_matrix(bml_type,bml_element_complex,dp,hdim,mdim,aux1_bml)
  call bml_zero_matrix(bml_type,bml_element_complex,dp,hdim,mdim,aux2_bml)

  call bml_convert_from_dense(bml_type,H,hmat_bml,numthresh,mdim)

  call bml_copy(hmat_bml,aux_bml)

  cf = cmplx(0.0_dp,2.0_dp)

  call bml_multiply(hmat_bml,hmat_bml,aux_bml,1.0_dp,0.0_dp,numthresh)

  call bml_scale(cf,aux_bml,aux1_bml)

  call bml_add(aux1_bml,hmat_bml,1.0_dp,-1.0_dp,numthresh)

  call bml_copy(aux1_bml, aux2_bml)

  call bml_convert_to_dense(aux2_bml,aux)

  write(*,*)"diagonal_bml = (",aux(1,1),aux(2,2),aux(3,3),"...)"

  !Non bml operations

  aux = 0.0_dp
  aux = -H + cf*(matmul(H,H))

  write(*,*)"diagonal_nbl = (",aux(1,1),aux(2,2),aux(3,3),"...)"

end program main
