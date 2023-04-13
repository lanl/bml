.. list-table::
  :header-rows: 1

  * - Issues
    - Pull Requests
    - CI
    - Conda
    - Documentation
    - Docker
  * - .. image:: https://img.shields.io/github/issues/lanl/bml
        :alt: GitHub issues
        :target: https://github.com/lanl/bml/issues>
    - .. image:: https://img.shields.io/github/issues-pr/lanl/bml.svg
        :alt: GitHub pull requests
        :target: https://github.com/lanl/bml/pulls
    - .. image:: https://github.com/lanl/bml/workflows/CI/badge.svg
        :alt: GitHub Actions
        :target: https://github.com/lanl/bml/actions
    - .. image:: https://anaconda.org/conda-forge/bml/badges/version.svg
        :alt: Conda Version
        :target: https://anaconda.org/conda-forge/bml
      .. image :: https://anaconda.org/conda-forge/bml/badges/downloads.svg
        :alt: Conda Downloads
        :target: https://anaconda.org/conda-forge/bml
    - .. image:: https://readthedocs.org/projects/basic-matrix-library/badge/?version=master
        :target: https://basic-matrix-library.readthedocs.io/en/master/?badge=master
        :alt: Documentation Status
    - .. image:: https://img.shields.io/docker/pulls/nicolasbock/bml
        :alt: Docker Pulls
        :target: https://hub.docker.com/repository/docker/nicolasbock/bml

Introduction
============

This website is intended to provide some guidance on how to get and install the
bml library. LA-UR number LA-UR-**17-27373**.

The basic matrix library (bml) is a collection of various matrix data formats
(for dense and sparse) and their associated algorithms for basic matrix
operations. Application programming interfaces (API) are available for both C
and FORTRAN. The current status of this library allows us to use two different
formats for representing matrix data. Currently these formats are: dense,
ELLPACK-R, ELLBLOCK, ELLSORT, and CSR. For information on how to use the BML
library can be find in
`BML-API <https://lanl.github.io/bml/API/developer_documentation.html>`_.

Mailing List
============

We are running the following mailing list for discussions on usage and features
of the bml library:

- `bml <https://groups.io/g/bml>`_
- `Subscribe <https://groups.io/g/bml/signup>`_
- `Archives <https://groups.io/g/bml/topics>`_

Supported Matrix Formats
========================

The bml library supports the following matrix formats:

- dense
- ELLPACK-R
- ELLSORT
- ELLBLOCK
- CSR

Binary Packages
===============

We offer binary packages of the bml library in `RPM format
<http://software.opensuse.org/download.html?project=home%3Anicolasbock%3Aqmmd&package=bml>`_
thanks to SUSE's OpenBuild Service and for Ubuntu in `DEB format
<https://launchpad.net/~nicolasbock/+archive/ubuntu/qmmd>`_.

Testing in our CI container
===========================

We are switching our CI tests from Travis-CI to GitHub Actions because Travis-CI
is `limiting the number of builds for open source projects
<https://blog.travis-ci.com/2020-11-02-travis-ci-new-billing>`_. Our workflow
uses a `custom Docker image <https://hub.docker.com/r/nicolasbock/bml>`_ which
comes with the necessary compiler tool chain to build and test the :code:`bml`
library. Using :code:`docker` is a convenient and quick way to develop, build,
and test the :code:`bml` library.

.. code-block:: console

  $ ./scripts/run-local-docker-container.sh
  latest: Pulling from nicolasbock/bml
  2f94e549220a: Already exists
  8d8ab0ffcd5e: Pull complete
  3fa4d3b6f5b4: Pull complete
  4f4fb700ef54: Pull complete
  Digest: sha256:18237f909f19896a57c658c93af5e8ed91c9fa596f15021be777a97444a3eaaf
  Status: Downloaded newer image for nicolasbock/bml:latest
  docker.io/nicolasbock/bml:latest
  groups: cannot find name for group ID 1000
  I have no name!@3a4ae718ba4f:/bml$

Inside the container:

.. code-block:: console

  I have no name!@6ea3f4937c0d:/bml$ ./build.sh compile
  Writing output to /bml/build.log
  Running command compile
  mkdir: created directory '/bml/build'
  mkdir: created directory '/bml/install'
  -- CMake version 3.12.1
  -- The C compiler identification is GNU 7.5.0
  -- The CXX compiler identification is GNU 7.5.0
  -- The Fortran compiler identification is GNU 7.5.0
  -- Check for working C compiler: /usr/bin/gcc
  -- Check for working C compiler: /usr/bin/gcc -- works

Alternatively, you can run one of the CI tests by executing e.g.

.. code-block:: console

  I have no name!@6ea3f4937c0d:/bml$ ./scripts/ci-gcc-10-C-single-real.sh
  +++ dirname ./scripts/ci-gcc-10-C-single-real.sh
  ++ readlink --canonicalize ./scripts/..
  + basedir=/bml
  + export CC=gcc-10
  + CC=gcc-10
  + export CXX=g++-11
  + CXX=g++-11
  + export FC=gfortran-11
  + FC=gfortran-11

Build Instructions
==================

The bml library is built with CMake. For convenience, we provide a shell script
which goes through the necessary motions and builds the library, runs the tests,
and installs it (in the :code:`install` directory).

For a quick installation
------------------------

We suggest to take a look at the :code:`example_build.sh` script that sets the
most important environmental variables needed by :code:`build.sh` script. Change
the Variables according to the compilers and architecture. The script can be run
just by doing:

.. code-block:: console

  $ ./scripts/example_build.sh
  Writing output to /bml/build.log
  Running command configure
  mkdir: created directory '/bml/build'
  mkdir: created directory '/bml/install'
  -- CMake version 3.12.1
  -- The C compiler identification is GNU 7.5.0
  -- The CXX compiler identification is GNU 7.5.0
  -- The Fortran compiler identification is GNU 7.5.0

For a more involved installation
--------------------------------

By running:

.. code-block:: console

  $ ./build.sh install

the library will be built in the :code:`build` directory and installed in the
:code:`install` directory. In case you change any sources and simply want to
rebuild the library, you don't have to run :code:`build.sh` again, but rather

.. code-block:: console

  $ make -C build

The compiled library can be installed by running

.. code-block:: console

  $ make -C build install

The install directory can be modified by running

.. code-block:: console

  $ CMAKE_INSTALL_PREFIX=/some/path ./build.sh install

(which assumes that you are using the bash shell).

To build with GNU compilers, OpenMP, and Intel MKL do the following.

.. code-block:: console

  $ CC=gcc FC=gfortran \
    BLAS_VENDOR=Intel CMAKE_BUILD_TYPE=Release \
    BML_OPENMP=yes CMAKE_INSTALL_PREFIX=/some/path \
    ./build.sh install

To build with MPI, OpenMP, and use Intel MKL do the following.

.. code-block:: console

  $ CC=mpicc FC=mpif90 \
    BLAS_VENDOR=Intel CMAKE_BUILD_TYPE=Release \
    BML_OPENMP=yes BML_MPI=yes CMAKE_INSTALL_PREFIX=/some/path \
    ./build.sh install

Prerequisites
-------------

In order to build the library, the following tools need to be installed:

- :code:`gcc` with Fortran support
- :code:`>=cmake-2.8.8`
- :code:`>=python-2.7`
- :code:`>=OpenMP-3.1` (i.e. :code:`>=gcc-4.7`)

If the build fails
------------------

In case the build fails for some reason, please contact the developers by
opening an issue on GitHub (https://github.com/lanl/bml/issues) and attach the
files

.. code-block:: shell

  build/CMakeFiles/CMakeOutput.log
  build/CMakeFiles/CMakeError.log

Developer Suggested Workflow
============================

Our main development happens on the :code:`master` branch and is continuously
verified for correctness. If you would like to contribute with your work to the
bml project, please follow the instructions at the GitHub help page `"About pull
requests" <https://help.github.com/articles/about-pull-requests/>`_. To
summarize:

- Fork the project on github
- Clone that forked repository
- Create a branch in it
- Commit any changes to the branch
- Push the branch to your forked repository
- Go to https://github.com/lanl/bml and click on 'Create Pull Request'

During the review process you might want to update your pull request. Please add
commits or :code:`amend` your existing commits as necessary. If you amend any
commits you need to add the :code:`--force-with-lease` option to the
:code:`git push` command. Please make sure that your pull request contains only
one logical change (see `"Structural split of change"
<https://wiki.openstack.org/wiki/GitCommitMessages#Structural_split_of_changes>`_
for further details.

Coding Style
============

Please indent your C code using

.. code-block:: console

  $ indent -gnu -nut -i4 -bli0 -cli4 -ppi0 -cbi0 -npcs -bfda

You can use the script :code:`indent.sh` to indent all C code.

Helpful Developer Resources
===========================

Optimizations
-------------

For low level optimization work it is useful to understand what assembly code
the compiler generates. For example, to verify that the compiler vectorizes the
loop in the following example:

.. code-block:: C
   :linenos:
   :lineno-start: 5
   :emphasize-lines: 4

   void double_array(float a[8]) {
     a = __builtin_assume_aligned(a, 64);
     for (int i = 0; i < 8; i++) {
      a[i] *= 2;
     }
   }

we can build the source with

.. code-block:: console

  gcc -S -O3 -fverbose-asm test.c

and analyze the generated assembly code,

.. code-block:: asm
   :linenos:
   :emphasize-lines: 2-4

   # test.c:8:    a[i] *= 2;
     movaps	(%rdi), %xmm0	# MEM <vector(4) float> [(float *)a_9], vect__5.8
     addps	%xmm0, %xmm0	#, vect__5.8
     movaps	%xmm0, (%rdi)	# vect__5.8, MEM <vector(4) float> [(float *)a_9]
     movaps	16(%rdi), %xmm0	# MEM <vector(4) float> [(float *)a_9 + 16B], vect__5.8
     addps	%xmm0, %xmm0	#, vect__5.8
     movaps	%xmm0, 16(%rdi)	# vect__5.8, MEM <vector(4) float> [(float *)a_9 + 16B]

The aligned memory access, `movaps`, moving 4 (aligned packed single-precision)
`float` values into `%xmm0`, and the subsequent `addps` instruction show that
the compiler fully vectorized the loop.

Note that the `Compiler Explorer <https://godbolt.org/>`_ provides an
alternative that does not require local compilations, see
`https://godbolt.org/z/ejEdqKa6Y <https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:___c,selection:(endColumn:1,endLineNumber:22,positionColumn:1,positionLineNumber:22,selectionStartColumn:1,selectionStartLineNumber:22,startColumn:1,startLineNumber:22),source:'%23include+%3Cstdio.h%3E%0A%0A%23define+N+8%0A%0Avoid+double_array(float+a%5BN%5D)+%7B%0A++a+%3D+__builtin_assume_aligned(a,+64)%3B%0A%23pragma+omp+simd%0A++for+(int+i+%3D+0%3B+i+%3C+N%3B+i%2B%2B)+%7B%0A+++a%5Bi%5D+*%3D+2%3B%0A++%7D%0A%7D%0A%0Aint+main+()+%7B%0A++float+a%5BN%5D+__attribute__((aligned(64)))%3B%0A++for+(int+i+%3D+0%3B+i+%3C+N%3B+i%2B%2B)+%7B%0A++++printf(%22a%5B%25d%5D+%3D+%25p%5Cn%22,+i,+%26a%5Bi%5D)%3B%0A++++a%5Bi%5D+%3D+i%3B%0A++%7D%0A++double_array(a)%3B%0A++printf(%22a%5B0%5D+%3D+%25e%5Cn%22,+a%5B0%5D)%3B%0A%7D%0A'),l:'5',n:'0',o:'C+source+%231',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:cg112,filters:(b:'0',binary:'1',commentOnly:'0',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:___c,libs:!(),options:'-O3',selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1,tree:'1'),l:'5',n:'0',o:'x86-64+gcc+11.2+(C,+Editor+%231,+Compiler+%231)',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4>`_.

Citing
======

If you find this library useful, we encourage you to cite us. Our project has a
citable DOI:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5570404.svg
   :target: https://doi.org/10.5281/zenodo.5570404

with the following :code:`bibtex` snipped:

.. code-block:: bibtex

  @misc{bml,
    author       = {Nicolas Bock and
                    Susan Mniszewski and
                    Bálint Aradi and
                    Michael Wall and
                    Christian F. A. Negre
                    Jamal Mohd-Yusof and
                    Anders N. M. Niklasson},
    title        = {qmmd/bml v2.1.2},
    month        = feb,
    year         = 2022,
    doi          = {10.5281/zenodo.5570404},
    url          = {https://doi.org/10.5281/zenodo.5570404}
  }

Another citation source is the following journal article (`DOI:
10.1007/s11227-018-2533-0 <https://doi.org/10.1007/s11227-018-2533-0>`_):

.. code-block:: bibtex

  @article{bock2018basic,
    title     = {The basic matrix library (BML) for quantum chemistry},
    author    = {Bock, Nicolas and
                 Negre, Christian FA and
                 Mniszewski, Susan M and
                 Mohd-Yusof, Jamaludin and
                 Aradi, B{\'a}lint and
                 Fattebert, Jean-Luc and
                 Osei-Kuffuor, Daniel and
                 Germann, Timothy C and
                 Niklasson, Anders MN},
    journal   = {The Journal of Supercomputing},
    volume    = {74},
    number    = {11},
    pages     = {6201--6219},
    year      = {2018},
    publisher = {Springer}
  }

Authors
=======

The core developers of the bml in alphabetical order:

- Christian Negre <cnegre@lanl.gov>
- Nicolas Bock <nicolasbock@gmail.com>
- Susan M. Mniszewski <smm@lanl.gov>

Contributors
============

- Adedoyin Adetokunbo <aadedoyin@lanl.gov>
- Bálint Aradi <aradi@uni-bremen.de>
- Daniel Osei-Kuffuor <oseikuffuor1@llnl.gov>
- Jamaludin Mohd-Yusof <jamal@lanl.gov>
- Jean-Luc Fattebert <fattebertj@ornl.gov>
- Mike Wall <mewall@lanl.gov>

License
=======

The bml library is licensed under the BSD 3-clause license.

Copyright 2015. Los Alamos National Security, LLC. This software was
produced under U.S. Government contract DE-AC52-06NA25396 for Los
Alamos National Laboratory (LANL), which is operated by Los Alamos
National Security, LLC for the U.S. Department of Energy. The
U.S. Government has rights to use, reproduce, and distribute this
software. NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
FOR THE USE OF THIS SOFTWARE. If software is modified to produce
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

LA-CC
=====

NOTICE OF OSS COPYRIGHT ASSERTION:

LANS has asserted copyright on the software package entitled *Basic
Matrix Library (bml), Version 0.x (C16006)*.

ABSTRACT
--------

The basic matrix library (bml) is a collection of various matrix data
formats (for dense and sparse) and their associated algorithms for basic
matrix operations.

This code is unclassified and has been assigned LA-CC-**15-093**. Los Alamos
National Laboratory’s Export Control Team made an in-house determination that
this software is controlled under Department of Commerce regulations and the
Export Control Classification Number (ECCN) **EAR99**. The export control
review is attached.

The developers intend to distribute this software package under the OSI
Certified **BSD 3-Clause License**
(http://www.opensource.org/licenses/BSD-3-Clause)

This code was developed using funding from:

- Basic Energy Sciences (LANL2014E8AN) and the Laboratory Directed Research
  and Development Program of Los Alamos National Laboratory. To tests these
  developments we used resources provided by the Los Alamos National
  Laboratory Institutional Computing Program, which is supported by the U.S.
  Department of Energy National Nuclear Security Administration

- Exascale Computing Project (17-SC-20-SC), a collaborative effort of two U.S.
  Department of Energy organizations (Office of Science and the National
  Nuclear Security Administration) responsible for the planning and
  preparation of a capable exascale ecosystem, including software,
  applications, hardware, advanced system engineering, and early testbed
  platforms, in support of the nation’s exascale computing imperative.

Larry Kwei, LAFO Program Manager, has granted his concurrence to asserting
copyright and then distributing the **Basic Matrix Library (bml), Version
0.x** code using an open source software license. See attached memo.

LANS acknowledges that it will comply with the DOE OSS policy as follows:

1. submit form DOE F 241.4 to the Energy Science and Technology Software
   Center (ESTSC),
2. provide the unique URL on the form for ESTSC to distribute, and
3. maintain an OSS Record available for inspection by DOE.

Following is a table briefly summarizes information for this software package:

.. list-table::

  * - CODE NAME
    - **Basic Matrix Library (bml), Version 0.x (C16006)**
  * - Classification Review Number
    - **LA-CC-15-093**
  * - Export Control Classification Number (ECCN)
    - **EAR99**
  * - B&R Code
    - **YN0100000**
