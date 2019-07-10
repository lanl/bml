#include "bml.h"
#include "../macros.h"
#include "../typed.h"

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <math.h>
#include <string.h>

// print content of file in stderr
void TYPED_FUNC(
    filecontent) (
    char *fname)
{
    FILE *fp = fopen(fname, "r");
    int ch;

    if (fp == NULL)
    {
        fprintf(stderr, "File %s doesn't exist\n", fname);
        return;
    }

    // Read contents of file
    while ((ch = fgetc(fp)) != EOF)
    {
        // print current character
        putc(ch, stderr);
    }

    fclose(fp);
}

int TYPED_FUNC(
    test_print) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    //generate random matrix
    bml_matrix_t *A = NULL;
    A = bml_random_matrix(matrix_type, matrix_precision, N, M, sequential);

    REAL_T *A_dense = bml_export_to_dense(A, dense_row_major);
    bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense, 0,
                           N, 0, N);

    // define filename for testing based on matrix_precision to avoid
    // conflicts when two tests are run simultaneously
    char filename[20];
    switch (matrix_precision)
    {
        case single_real:
            strcpy(filename, "single");
            break;
        case double_real:
            strcpy(filename, "double");
            break;
        case single_complex:
            strcpy(filename, "scomplex");
            break;
        case double_complex:
            strcpy(filename, "dcomplex");
            break;
        default:
            fprintf(stderr, "Unknown precision\n");
            break;
    }
    char str[10] = "_test.dat";
    strcat(filename, str);
    fprintf(stdout, "filename used for this test: %s\n", filename);

    // Redirect stdout
    // note: no good way to restore it later, so don't try to print
    // anything else but the matrix
    FILE *fp = freopen(filename, "w", stdout);
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to associate %s with stdout", filename);
        return -1;
    }

    //assumes matrix is at least 2x2
    const int up = 2;
    bml_print_bml_matrix(A, 0, up, 0, up);

    if (fclose(fp) == EOF)
    {
        fprintf(stderr, "ERROR closing file1\n");
        return -1;
    }

    fprintf(stderr, "FILE content:\n");
    TYPED_FUNC(filecontent) (filename);

    //now read file just written
    REAL_T *data = calloc(up * up, sizeof(REAL_T));
    if (data == NULL)
    {
        fprintf(stderr, "callo failed!\n");
        return -1;
    }

    FILE *fp2 = fopen(filename, "r");
    if (fp2 == NULL)
    {
        fprintf(stderr, "Failed to open %s\n", filename);
        return -1;
    }

    //FILE *fp3 = fopen("test_output.dat", "w");

    float realp;
    float imagp;
    char sign;

    for (int i = 0; i < up; i++)
    {
        for (int j = 0; j < up; j++)
        {
            switch (matrix_precision)
            {
                case single_real:
                case double_real:
                    fscanf(fp2, "%f", &realp);
                    data[ROWMAJOR(i, j, up, up)] = realp;
                    break;
                case single_complex:
                case double_complex:
                    //read complex number in 3 parts, discarding 'i'
                    fscanf(fp2, "%f %c %fi", &realp, &sign, &imagp);
                    if (sign == '-')
                        imagp *= -1;
                    REAL_T tmp = realp + imagp * _Complex_I;
                    //fprintf(fp3, "tmp1 %f \n", REAL_PART(tmp));
                    //fprintf(fp3, "tmp1 %f \n", IMAGINARY_PART(tmp));
                    data[ROWMAJOR(i, j, up, up)] = tmp;

                    break;
                default:
                    fprintf(stderr, "Unknown precision\n");
                    break;
            }
        }
    }
    if (fclose(fp2) == EOF)
    {
        fprintf(stderr, "ERROR closing file2\n");
        return -1;
    }

    //compare data just read with original data
    const double tol = 1.e-3;
    for (int i = 0; i < up; i++)
    {
        for (int j = 0; j < up; j++)
        {
            REAL_T val1 = data[ROWMAJOR(i, j, up, up)];
            REAL_T val2 = A_dense[ROWMAJOR(i, j, N, M)];
            double diff1 = REAL_PART(val2 - val1);
            double diff2 = IMAGINARY_PART(val2 - val1);
            double diff = sqrt(diff1 * diff1 + diff2 * diff2);
//            if (diff > tol)
            {
                fprintf(stderr, "real parts= %f and %f\n", REAL_PART(val1),
                        REAL_PART(val2));
                fprintf(stderr, "imag part= %f and %f\n",
                        IMAGINARY_PART(val1), IMAGINARY_PART(val2));
                fprintf(stderr, "i=%d, j=%d, diff=%lf\n", i, j, diff);
                //fclose(fp3);
                if (diff > tol)
                {
                    fprintf(stderr, "test failed!!!\n");
                    return -1;
                }
            }
        }
    }
    //fprintf(fp3, "Test Successful!\n");
    //fclose(fp3);

    bml_deallocate(&A);
    free(data);

    if (remove(filename) != 0)
    {
        fprintf(stderr, "Failed removing file %s\n", filename);
        return -1;
    }

    return 0;
}
