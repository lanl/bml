ADDING A TEST
=============

It is essential to add a proper test for each function we create. We would even
recommend to add a test before adding the functionality to have a piece of code
that could be executed. To do this, we have provided this step-by-step tutorial.
Let's consider that we are adding a test which name is "mytest".

We will first modify the three following files accordingly by adding the name of
the test in them. Note: Whenever we can we will proceed to add names/files in
alphabetical order to keep consistency in the source file.

The three files that need to be modified are:

  - /tests/CMakeLists.txt
  - /tests/bml_test.c
  - /tests/bml_test.h

In CMakeLists.txt we will add the test name in three places:

  	set(SOURCES_TYPED
     	test1_typed.c
     	...
    	mytest_typed.c
     	...
     	testN_typed.c)

;

  	add_executable(bml-test
     	test1.c
    	...
    	mytest.c
    	...
    	testN.c)

and

  	foreach(N add test1 ... mytest ... testN)

Second, we should modify the bml_test.h to include our "future" header file.
We will add the name as follows:

	#include "test1.h"
	...
	#include "mytest.h"
	...
	#include "testN.h"


Finally, we will modify the bml_test.c file in four positions. We will first
indicate that there is going to be an extra test by increasing the NUM_TEST
variable:

	const int NUM_TESTS = <N>;

where N has to be replace by the total number of tests. Next we will add the
test name in the test_name array:

	const char *test_name[] =
 			{ "test1", ... , "mytest", ... , "testN"}

Please ensure that the number of entries in test_name, test_description, and
testers matches the value of NUM_TEST.
This will be followed by a description of the test:

	const char *test_description[] = {
			 "Description of test 1",
			  ....
			 "Description of mytest",
			  ....
			 "Description of test N"}

And finally we will add the name of the function that will perform the test:

	const test_function_t testers[] = {
			    test_test1,
 			    ...
 			   test_mytest,
 			    ...
 			   test_testN}


After this is done we will start creating the source code for our test.
These files will be created inside /tests/ and will be named as follows:

- /tests/mytest.c
- /tests/mytest.h
- /tests/mytest_typed.c

This means that for each test we will have a "header file" (mytest.h), a
"driver" (mytest.c) and a typed (mytest_typed.c) . In this last file we will
add all the fuctionalities for testing (actual test). For these three files
we provide templates which names are template.c, template.h and
template_typed.c . These files (template-) will have to be renamed to (mytest-).
The final step which is left to the developer is to add some lines of code
inside mytest_typed.c to make the test work. For example, this can be a
difference between two values that has to be less than a tolerance.


## Compiling, running and checking the test

Once the functionality is added we need to make sure that the test is compiling,
running and passing. For this we can do the following:

First we can try to configure the code using the example_build.sh file located
inside the main directory. Second, if the configuration proceeds with no error
we build the code:

	$ ./example_build
	$ cd build; make

If everything is built without problems. We can test the whole code:

	$ make test

or if we want to see details of the test:

	$ make test ARGS="-V"

We can check if the new test we have added appears in the list of tests.

If we want to run just the test we have created we can do:

	$ cd /build/tests
	$ ./bml-test -n mytest -t ellpack -p double_complex

The latter means that we will run our test with ellpack matrix type and
double_complex precision. Once the test passes for every precision and matrix
type we will need to make sure there are no memory leaks in the test or routine.
For this we could run valgrind as following:

	$ valgrind ./bml-test -n mytest -t ellpack -p double_complex

You can also trigger tests by running ctest directly.

  $ cd build
  $ ctest -R mytest --output-on-failure

After all the tests passed, we should indent the new files using the indent.sh
Running indent.sh (located in the main folder) will indent all files.

	$ ./indent.sh
