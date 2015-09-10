#include "bml.h"
#include "bml_test.h"

#define STRINGIFY(a) #a

int main(int argc, char **argv)
{
    const int N = 7;

    bml_log(BML_INFO, "testing %s\n", STRINGIFY(MATRIX_TYPE_NAME));
    if(test_function(N, MATRIX_TYPE_NAME, MATRIX_PRECISION) != 0) {
        bml_log(BML_ERROR, "test failed\n");
    }
    bml_log(BML_INFO, "test passed\n");
}
