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

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

//unsigned int
//gcd(unsigned int a, unsigned int b)
//{
//    unsigned int x;
//    while (b)
//    {
//        x = a % b;
//        a = b;
//        b = x;
//    }
//    return a;
//}

/*
 *  * The binary gcd algorithm using iteration.
 *   * Should be fairly fast.
 *    *
 *     * Put in the public domain by the author:
 *      *
 *       * Christian Stigen Larsen
 *        * http://csl.sublevel3.org
 *         */
//int gcd(int u, int v)
//{
//    int shl = 0;
//
//    while ( u && v && u!=v ) {
//        int eu = !(u & 1);
//        int ev = !(v & 1);
//
//        if ( eu && ev ) {
//            ++shl;
//            u >>= 1;
//            v >>= 1;
//        }
//        else if ( eu && !ev ) u >>= 1;
//        else if ( !eu && ev ) v >>= 1;
//        else if ( u>=v ) u = (u-v)>>1;
//        else {
//            int tmp = u;
//            u = (v-u)>>1;
//            v = tmp;
//        }
//    }
//
//    return !u? v<<shl : u<<shl;
//}

int gcd(int u,int v)
{
    int k=0,t=0,i;
    while (!(u&1) && !(v&1))
    {
    k++;
    u>>=1;
    v>>=1;
    }
    if (u&1)
    t=u;
    else
    t=-v;
    do
    {
    while (!(t&1))
    t>>=1;
    if (t>0)
    u=t;
    else
    v=-t;
    t=u-v;
    }while (t);
    for (i=0;i<k;i++)
    u<<=1;
    return(u);
}

typedef struct
{
    int start, stop, number_of_threads, result;
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
    if ((value.start < 0) ||
        (value.stop  < 0) ||
        (value.stop < value.start))
    {
        value.start = 0;
        value.stop  = 0;
    }

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
    const int number_of_input_sets = read_integer();

    int* gcd_lookup = malloc(sizeof(int) * 200 * 200);

    for (int n = 1; n < 200; ++n)
    {
        for (int m = n + 1; m < 200; m += 2)
        {
            gcd_lookup[m * 200 + n] = gcd(m, n);
        }
    }

    if (number_of_input_sets > 0)
    {
        input_set input_sets[number_of_input_sets];

        for (int i = 0; i < number_of_input_sets; ++i)
        {
            input_sets[i] = read_input_set();
        }

        for (int i = 0; i < number_of_input_sets; ++i)
        {
            int number_of_pythagorean_triplets = 0;

            const int upper_boundary = (int)ceil(sqrt(input_sets[i].stop));

//            #pragma omp parallel for reduction(+: number_of_pythagorean_triplets)
//                                     num_threads(input_sets[i].number_of_threads)
//                                     schedule(static)
            for (int n = 1; n < upper_boundary; ++n)
            {
                const int nn = n * n;

                // m is incremented by 2 for each iteration such
                // that (m - n) is always odd.

                int lower_boundary = n + 1;

                if (nn < input_sets[i].start)
                {
                    int wtf = (int)floor(sqrt(input_sets[i].start - nn));
                    wtf += (int)!((wtf - n) & 1);
                    if (wtf > lower_boundary)
                    {
                        lower_boundary = wtf;
                    }
                }

                for (int m = lower_boundary; m < upper_boundary; m += 2)
                {
                    int g = gcd_lookup[m * 200 + n];
                    //if (gcd(m, n) == 1)
                    if (g == 1)
                    {
                        const int mm = m * m;
                        const int c = mm + nn;

                        ///if (c >= input_sets[i].stop)
                        ///{
                        ///    break;
                        ///}

                        ///if (c >= input_sets[i].start)
                        ///{
                        ///    number_of_pythagorean_triplets++;
                        ///}

                        if (c >= input_sets[i].start && c < input_sets[i].stop)
                        {
                            number_of_pythagorean_triplets++;
                        }
                    }
                }
            }

            input_sets[i].result = number_of_pythagorean_triplets;
        }

        for (int i = 0; i < number_of_input_sets; ++i)
        {
            printf("%d\n", input_sets[i].result);
        }
    }

    return EXIT_SUCCESS;
}

