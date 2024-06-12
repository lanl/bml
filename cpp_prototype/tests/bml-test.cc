#include <bml.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
    printf("starting tests\n");
    BMLMatrix * A = bml_identity_matrix(dense, double_real, 10, 10, sequential);
}
