# Build Instructions #

The bml library is built with CMake.  For your convenience, we provide
a shell script which goes through the necessary motions and builds the
library, runs the tests, and installs it (in the `install` directory).
Simply run:

    $ ./full-test.sh

# Developer Suggested Workflow #

We try to preserve a linear history in our main (master)
branch. Instead of pulling, we suggest you use:

    $ git pull --rebase=preserve

And then

    $ git push

To push your changes back to the server.
