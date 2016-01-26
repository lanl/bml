#define _POSIX_C_SOURCE 200809L

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

const int NUM_TESTS = 1;
const char *test_name[] = { "new" };

void
print_usage(
    void)
{
    printf("Usage:\n");
    printf("\n");
    printf("-h | --help       This help\n");
    printf("-t | --test TEST  Run test TEST\n");
    printf("-l | --list       List all available tests\n");
}

int
main(
    int argc,
    char **argv)
{
    const char *short_option = "ht:l";
    const struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {"test", required_argument, NULL, 't'},
        {"list", no_argument, NULL, 'l'},
        {NULL, 0, NULL, 0}
    };
    int c;

    while ((c =
            getopt_long(argc, argv, short_option, long_options, NULL)) != -1)
    {
        switch (c)
        {
            case 'h':
                print_usage();
                return 0;
                break;
            case 't':
            {
                char *test_str = strdup(optarg);
                int i;
                for (i = 0; i < NUM_TESTS; i++)
                {
                    if (strcasecmp(test_str, test_name[i]) == 0)
                    {
                        break;
                    }
                }
                if (i == NUM_TESTS)
                {
                    fprintf(stderr, "unknown test %s\n", test_str);
                    return 1;
                }
                break;
            }
            case 'l':
                for (int i = 0; i < NUM_TESTS; i++)
                {
                    printf("%s", test_name[i]);
                    if (i < NUM_TESTS - 1)
                    {
                        printf(", ");
                    }
                }
                printf("\n");
                break;
        }
    }

    return 0;
}
