#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//#ifdef HAVE_MPI
//#include <mpi.h>
//#endif
//
//#ifdef HAVE_OPENMP
//#include <omp.h>
//#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

int
gcd(int a, int b)
{
    int x;
    while (b)
    {
        x = a % b;
        a = b;
        b = x;
    }
    return a;
}

typedef struct
{
    int start, stop, number_of_threads;
} input_set;

input_set
read_input_set()
{
    input_set value = {};

    char* line_string = NULL;
    size_t line_length;

    if (getline(&line_string, &line_length, stdin) == -1)
    {
        fprintf(stderr, "read_input_set(): getline() failure\n");
        exit(EXIT_FAILURE);
    }

    if (sscanf(line_string,
               "%d %d %d",
               &value.start,
               &value.stop,
               &value.number_of_threads) < 2)
    {
        fprintf(stderr, "read_input_set(): sscanf() failure\n");
        exit(EXIT_FAILURE);
    }

    free(line_string);

    return value;
}

int
read_integer()
{
    int value = 0;

    char* line_string = NULL;
    size_t line_length;

	if (getline(&line_string, &line_length, stdin) == -1)
    {
        fprintf(stderr, "read_integer(): getline() failure\n");
        exit(EXIT_FAILURE);
    }

    if (sscanf(line_string, "%d", &value) != 1)
    {
        fprintf(stderr, "read_integer(): sscanf() failure\n");
        exit(EXIT_FAILURE);
    }

    free(line_string);

    return value;
}

int
main(const int argc, char** const argv)
{
//#if defined(_OPENMP)
//#pragma omp parallel
//    {
//
//    int NCPU,tid,NPR,NTHR;
//    /* get the total number of CPUs/cores available for OpenMP */
//    NCPU = omp_get_num_procs();
//    /* get the current thread ID in the parallel region */
//    tid = omp_get_thread_num();
//    /* get the total number of threads available in this parallel region */
//    NPR = omp_get_num_threads();
//    /* get the total number of threads requested */
//    NTHR = omp_get_max_threads();
//    /* only execute this on the master thread! */
//    if (tid == 0) {
//        printf("%i : NCPU\t= %i\n",tid,NCPU);
//        printf("%i : NTHR\t= %i\n",tid,NTHR);
//        printf("%i : NPR\t= %i\n",tid,NPR);
//    }
//    printf("%i : hello multicore user! I am thread %i out of %i\n",tid,tid,NPR);
//    }
//
//    return 0;
//#endif


    const int number_of_input_sets = read_integer();

    input_set input_sets[number_of_input_sets];

    for (int i = 0; i != number_of_input_sets; ++i)
    {
        input_sets[i] = read_input_set();

        int number_of_pythagorean_triplets = 0;

        for (int n = 1; n < input_sets[i].stop; ++n)
        {
            const int nn = n * n;

            // m is incremented by 2 for each iteration such
            // that (m - n) is always odd.

            #pragma omp parallel for reduction(+: number_of_pythagorean_triplets)
            for (int m = n + 1; m < input_sets[i].stop; m += 2)
            {
                if (gcd(m, n) == 1)
                {
                    const int mm = m * m;
                    const int c  = mm + nn;

                    if (c >= input_sets[i].start && c < input_sets[i].stop)
                    {
                        number_of_pythagorean_triplets++;
                    }
                }
            }
        }

        printf("%d\n", number_of_pythagorean_triplets);
    }

    return EXIT_SUCCESS;
}

