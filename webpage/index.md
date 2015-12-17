---
title: BML
---

# Build Instructions #

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

## Prerequisites ##

In order to build the library, the following tools need to be installed:

- `gcc` with Fortran support
- `>=cmake-2.8.8`
- `>=python-2.7`

## If the build fails ##

In case the build fails for some reason, please email the developers
at <qmmd-all@lanl.gov> or open an issue on github
(https://github.com/qmmd/bml/issues) and attach the files

    build/CMakeFiles/CMakeOutput.log
    build/CMakeFiles/CMakeError.log

# Developer Suggested Workflow #

We do our main development on the `develop` branch.  If you would like to
contribute your work to the bml project, please fork the project on the GitHub
webpage. Replace `USERNAME` in the following with your GitHub username.

~~~
$ git clone git@github.com:USERNAME/bml.git
$ git remote add upstream git@github.com:qmmd/bml.git
$ git fetch --all
$ git checkout -b feature origin/develop
~~~

You should name the branch something more exciting than simply `feature` to
indicate better what it is for. Now work on the branch and commit as often as
you like. When you are done and think you want to push your changes back to
GitHub for us to have a look at, run

~~~
$ git push --set-upstream origin feature
~~~

Open a new pull request on the GitHub webpage and make sure to set `base-fork:
qmmd/bml` and `base: develop`.

You can find a good description of how this works
[here](https://help.github.com/articles/using-pull-requests/).

## Coding Style ##

Please indent your C code using

    $ indent -gnu -nut -i4 -bli0 -cli4 -ppi0 -cbi0 -npcs -bfda

You can use the script `indent.sh` to indent all C code.

# License #

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

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of Los Alamos National Security, LLC, Los Alamos National
  Laboratory, LANL, the U.S. Government, nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission

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
