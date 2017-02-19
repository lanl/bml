---
title: BML
---

| master | develop |
| ------ | ------- |
| [![Build Status](https://travis-ci.org/qmmd/bml.svg?branch=master)](https://travis-ci.org/qmmd/bml) | [![Build Status](https://travis-ci.org/qmmd/bml.svg?branch=develop)](https://travis-ci.org/qmmd/bml) |
| [![codecov.io](https://codecov.io/github/qmmd/bml/coverage.svg?branch=master)](https://codecov.io/github/qmmd/bml?branch=master) | [![codecov.io](https://codecov.io/github/qmmd/bml/coverage.svg?branch=develop)](https://codecov.io/github/qmmd/bml?branch=develop) |

# Build Instructions

The bml library is built with CMake.  For your convenience, we provide
a shell script which goes through the necessary motions and builds the
library, runs the tests, and installs it (in the `install` directory).
Simply run:

    $ ./build.sh compile

and the library will be built in the `build` directory.  In case you
change any sources and simply want to rebuild the library, you don't
have to run `build.sh` again, but rather

    $ make -C build

The compiled library can be installed by running

    $ make -C build install

which by default installs in `/usr/local`.  The install directory can
be modified by running

    $ CMAKE_INSTALL_PREFIX=/some/path ./build.sh configure

(which assumes that you are using the bash shell).

To build with GNU compilers, OpenMP, and Intel MKL do the following.

    $ CC=gcc FC=gfortran BLAS_VENDOR=Intel CMAKE_BUILD_TYPE=Release BML_OPENMP=yes CMAKE_INSTALL_PREFIX=/some/path ./build.sh configure

To build with MPI, OpenMP, and use Intel MKL do the following.

    $ CC=mpicc FC=mpif90 BLAS_VENDOR=Intel CMAKE_BUILD_TYPE=Release BML_OPENMP=yes BML_MPI=yes CMAKE_INSTALL_PREFIX=/some/path ./build.sh configure

## Prerequisites

In order to build the library, the following tools need to be installed:

- `gcc` with Fortran support
- `>=cmake-2.8.8`
- `>=python-2.7`
- `>=OpenMP-3.1` (i.e. `>=gcc-4.7)

## If the build fails

In case the build fails for some reason, please contact the developers by
opening an issue on GitHub (https://github.com/qmmd/bml/issues) and attach the
files

    build/CMakeFiles/CMakeOutput.log
    build/CMakeFiles/CMakeError.log

# Developer Suggested Workflow

We do our main development on the `develop` branch.  If you would like to
contribute your work to the bml project, please fork the project on github,
hack away at the forked `develop` branch and send us a pull request once you
think we should have a look and integrate your work.

## Coding Style

Please indent your C code using

    $ indent -gnu -nut -i4 -bli0 -cli4 -ppi0 -cbi0 -npcs -bfda

You can use the script `indent.sh` to indent all C code.

# Citing

If you find this library useful, we encourage you to cite us. Our project has
a citable DOI:

[![DOI](https://zenodo.org/badge/20454/qmmd/bml.svg)](https://zenodo.org/badge/latestdoi/20454/qmmd/bml)

with the following `bibtex` snipped:

    @misc{bml,
      url = {\url{https://qmmd.github.io}}
      author = {Aradi, B\'{a}lint and Bock, Nicolas and Mniszewski, Susan M.
        and Mohd-Yusof, Jamaludin and Negre, Christian},
      year = 2017
    }

# License

The bml library is licensed under the BSD 3-clause license.

Copyright 2015. Los Alamos National Security, LLC. This software was
produced under U.S. Government contract DE-AC52-06NA25396 for Los
Alamos National Laboratory (LANL), which is operated by Los Alamos
National Security, LLC for the U.S. Department of Energy. The
U.S. Government has rights to use, reproduce, and distribute this
software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
derivative works, such modified software should be clearly marked, so
as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:
- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
- Neither the name of Los Alamos National Security, LLC, Los Alamos
  National Laboratory, LANL, the U.S. Government, nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS
ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# LA-CC

NOTICE OF OSS COPYRIGHT ASSERTION:

LANS has asserted copyright on the software package entitled *Basic
Matrix Library (bml), Version 0.x (C16006)*.

## ABSTRACT

The basic matrix library (bml) is a collection of various matrix data
formats (in dense and sparse) and their associated algorithms for basic
matrix operations.

This code is unclassified and has been assigned LA-CC-**15-093**. Los Alamos
National Laboratory’s Export Control Team made an in-house determination that
this software is controlled under Department of Commerce regulations and the
Export Control Classification Number (ECCN) **EAR99**. The export control
review is attached.

The developers intend to distribute this software package under the OSI
Certified **BSD 3-Clause License**
(http://www.opensource.org/licenses/BSD-3-Clause)

This code was developed using funding from the LANL Laboratory-Directed
Research Development (LDRD) Program. Larry Kwei, LAFO Program Manager, has
granted his concurrence to asserting copyright and then distributing the
**Basic Matrix Library (bml), Version 0.x** code using an open source software
license. See attached memo.

LANS acknowledges that it will comply with the DOE OSS policy as follows:

1. submit form DOE F 241.4 to the Energy Science and Technology Software
   Center (ESTSC),
2. provide the unique URL on the form for ESTSC to distribute, and
3. maintain an OSS Record available for inspection by DOE.

Following is a table briefly summarizes information for this software package:

| CODE NAME                                   | Basic Matrix Library (bml), Version 0.x (C16006) |
| ------------------------------------------- | ------------------------------------------------ |
| Classification Review Number                | **LA-CC-15-093**                                 |
| Export Control Classification Number (ECCN) | **EAR99**                                        |
| B&R Code                                    | **YN0100000**                                    |
