# Build Instructions #

The bml library is built with CMake.  For your convenience, we provide
a shell script which goes through the necessary motions and builds the
library, runs the tests, and installs it (in the `install` directory).
Simply run:

    $ ./build.sh

## If the build fails ##

In case the build fails for some reason, please email the developers
at <qmmd-all@lanl.gov> and attach the files

    build/CMakeFiles/CMakeOutput.log
    build/CMakeFiles/CMakeError.log

# Developer Suggested Workflow #

We try to preserve a linear history in our main (master)
branch. Instead of pulling (i.e. merging), we suggest you use:

    $ git pull --rebase

And then

    $ git push

To push your changes back to the server.

## Coding Style ##

Please indent your C code using

    $ indent -gnu -nut -i4 -bli0
