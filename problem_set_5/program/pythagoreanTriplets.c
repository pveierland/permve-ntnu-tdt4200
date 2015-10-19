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

//#if defined(_OPENMP)
//#include <omp.h>
//#endif

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

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

    if ((getline(&line_string, &line_length, stdin) == -1) ||
        (sscanf(line_string,
                "%d %d %d",
                &value.start,
                &value.stop,
                &value.number_of_threads) < 2))
    {
        value.start             = 0;
        value.stop              = 0;
        value.number_of_threads = 1;
    }

    // Sanitize input values
    value.start             = max(value.start, 0);
    value.stop              = max(value.stop, value.start);
    value.number_of_threads = max(value.number_of_threads, 1);

    free(line_string);

    return value;
}

int
read_integer()
{
    int value = 0;

    char* line_string = NULL;
    size_t line_length;

	if ((getline(&line_string, &line_length, stdin) == -1) ||
        (sscanf(line_string, "%d", &value) != 1))
    {
        value = 0;
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

    if (number_of_input_sets > 0)
    {
        input_set input_sets[number_of_input_sets];

        for (int i = 0; i < number_of_input_sets; ++i)
        {
            input_sets[i] = read_input_set();

            int number_of_pythagorean_triplets = 0;

            //const int upper_boundary = (int)ceil(sqrt(input_sets[i].stop));
            const int upper_boundary = input_sets[i].stop;

            for (int n = 1; n < upper_boundary; ++n)
            {
                const int nn = n * n;

                // m is incremented by 2 for each iteration such
                // that (m - n) is always odd.

                //int lower_boundary = n + 1;
                //if (nn < input_sets[i].start)
                //{
                //    lower_boundary = (int)ceil(sqrt(input_sets[i].start - nn));

//                #ifdef HAVE_OPENMP
//                #pragma omp parallel for reduction(+: number_of_pythagorean_triplets)
//                #endif
                for (int m = n + 1; m < upper_boundary; m += 2)
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
    }

    return EXIT_SUCCESS;
}

