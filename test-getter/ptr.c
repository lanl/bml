#include <bml.h>
#include <stdio.h>
#include <stdlib.h>

int
main(
    int argc,
    char **argv)
{
    bml_matrix_t *A;
    float *diagonal;
    float *row;
    float *Aij;

    A = bml_random_matrix(ellpack, single_real, 10, 10, sequential);
    //A = bml_identity_matrix(ellpack, single_real, 10, 10, sequential);
    diagonal = bml_get_diagonal(A);
    row = bml_get_row(A, 0);
    Aij = bml_get(A, 0, 0);

    bml_print_bml_matrix(A, 0, 10, 0, 10);

    printf("diagonal:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%f\n", diagonal[i]);
    }
    free(diagonal);

    printf("row[0]\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%f\n", row[i]);
    }
    free(row);

    printf("A[0][0] = %f\n", *Aij);

    bml_deallocate(&A);
}
