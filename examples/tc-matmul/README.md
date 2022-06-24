SP2 Tensor Core Version
=======================

# To compile #

Load the following modules. We will hereby take the Darwin LANL cluster 
as a reference, with a power9 node allocation. Load the modules 
as follows:
```bash 
module load cuda gcc cmake 
```

Add the following line to the example_build.sh script 
on the main source file. 

For C++ callable routine:
```bash 
export PROGRESS_SP2TC=${PROGRESS_SP2TC:=C++} 
```

For generating a fortran library:
```bash 
export PROGRESS_SP2TC=${PROGRESS_SP2TC:=Fortran} 
```

Run the build script and compile the code as follows: 
```bash 
./example_build.sh 
cd build; make; make install 
```
# To run and test # 
There are two examples located in `/examples`: A C++ 
example in `sp2tc_C++`, and a Fortran exmple in 
sp2tc_Fortran.
 



