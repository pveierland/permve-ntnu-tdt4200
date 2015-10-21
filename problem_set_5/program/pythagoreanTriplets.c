#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//#ifdef HAVE_MPI
//#include <mpi.h>
//#endif
//
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

static inline
int
calculate_lower_m_boundary(const int start, const int n, const int nn)
{
    int lower_m_boundary = max(n + 1, (int)ceil(sqrt(start - nn)));
    lower_m_boundary += (int)!((lower_m_boundary - n) & 1);
    return lower_m_boundary;
}

static inline
int
calculate_upper_m_boundary(const int stop, const int nn)
{
    return (int)ceil(sqrt(stop - nn));
}

static inline
int
calculate_upper_n_boundary(const int stop)
{
    return (int)ceil((sqrt(2 * stop - 1) - 1.0) / 2.0);
}

unsigned int
gcd(unsigned int a, unsigned int b)
{
    unsigned int x;
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
        value.start             = 1;
        value.stop              = 1;
        value.number_of_threads = 1;
    }

    // Sanitize input values
    if ((value.start < 0) ||
        (value.stop  < 0) ||
        (value.stop  < value.start))
    {
        value.start = 1;
        value.stop  = 1;
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

    if (number_of_input_sets > 0)
    {
        input_set input_sets[number_of_input_sets];

        for (int i = 0; i < number_of_input_sets; ++i)
        {
            input_sets[i] = read_input_set();
        }

        for (int i = 0; i < number_of_input_sets; ++i)
        {
            const int start = input_sets[i].start;
            const int stop  = input_sets[i].stop;

            int number_of_pythagorean_triplets = 0;

            const int upper_n_boundary = calculate_upper_n_boundary(stop);

            #ifdef HAVE_OPENMP
            #pragma omp parallel for reduction(+: number_of_pythagorean_triplets) \
                                     num_threads(input_sets[i].number_of_threads)
            #endif
            for (int n = 1; n < upper_n_boundary; ++n)
            {
                const int nn               = n * n;
                const int lower_m_boundary = calculate_lower_m_boundary(start, n, nn);
                const int upper_m_boundary = calculate_upper_m_boundary(stop, nn);

                // m is incremented by 2 for each iteration such
                // that (m - n) is always odd.

                for (int m = lower_m_boundary; m < upper_m_boundary; m += 2)
                {
                    if (gcd(m, n) == 1)
                    {
                        number_of_pythagorean_triplets++;
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

