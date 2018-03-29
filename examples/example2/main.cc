extern "C"
{
#include <bml_allocate.h>
#include <bml_setters.h>
#include <bml_copy.h>
#include <bml_multiply.h>
#include <bml_utilities.h>
}

int
main(
    )
{
    const int n = 10;           //matrix size
    const int m = 2;            //number of nonzero/row
    bml_matrix_t *A = bml_zero_matrix(ellpack, double_real, n, m, sequential);

    // initialize matrix A
    const double dval = 2.;
    for (int i = 0; i < n; i++)
        bml_set_element(A, i, i, &dval);
    const double oval = -1.;
    for (int i = 0; i < n - 1; i++)
        bml_set_element(A, i + 1, i, &oval);

    // initialize matrix B
    bml_matrix_t *B = bml_copy_new(A);

    // compute C=A*B
    bml_matrix_t *C = bml_zero_matrix(ellpack, double_real, n, n, sequential);
    const double threshold = 1.e-6;
    bml_multiply_AB(A, B, C, threshold);

    // print result
    bml_print_bml_matrix(C, 0, n - 1, 0, n - 1);

    bml_deallocate(&A);
    bml_deallocate(&B);
    bml_deallocate(&C);

    return 0;
}
