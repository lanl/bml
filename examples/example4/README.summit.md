To run in Summit: 
================

Log into a node: 

	bsub -W 60 -nnodes 1 -alloc_flags "NVME" -P MAT187 -Is /bin/bash

Export magma lib path: 

	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/magma-2.5.0-orig/lib

Run using jsrun: 
	
	export OMP_NUM_THREADS=21

	jsrun -n1 -a1 -g1 -c21 -bpacked:21 time ./main


