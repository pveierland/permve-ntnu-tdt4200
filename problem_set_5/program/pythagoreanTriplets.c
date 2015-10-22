#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define checked_malloc(size) assert_malloc(size, __FILE__, __LINE__)

void*
assert_malloc(const size_t size, const char* const file, const int line)
{
    void* v = malloc(size);
    if (!v)
    {
        fprintf(stderr, "error: %s:%d malloc(%zu) failed!\n", file, line, size);

        #ifdef HAVE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
        #else
        exit(-1);
        #endif
    }
    return v;
}

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
    int start, stop, number_of_threads;
} input_set;

input_set
read_input_set()
{
    input_set value = {0};

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
    else
    {
        value.start = max(value.start, 1);
        value.stop  = max(value.stop, 1);
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
    (void)argc; (void)argv;

    #ifdef HAVE_MPI
    MPI_Init(NULL, NULL);
    #endif

    int world_rank = 0;
    int world_size = 1;

    #ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    #endif

    int number_of_input_sets;

    if (world_rank == 0)
    {
        number_of_input_sets = read_integer();
    }
    
    #ifdef HAVE_MPI
    MPI_Bcast(&number_of_input_sets, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #endif

    input_set* const input_sets = checked_malloc(number_of_input_sets * sizeof(input_set));

    if (world_rank == 0)
    {
        for (int i = 0; i < number_of_input_sets; ++i)
        {
            input_sets[i] = read_input_set();
        }
    }

    #ifdef HAVE_MPI
    MPI_Bcast(input_sets,
              number_of_input_sets * sizeof(input_set) / sizeof(int),
              MPI_INT,
              0,
              MPI_COMM_WORLD);
    #endif

    int* const results = checked_malloc(number_of_input_sets * sizeof(int));

    for (int i = 0; i < number_of_input_sets; ++i)
    {
        const int start = input_sets[i].start;
        const int stop  = input_sets[i].stop;

        const int upper_n_boundary = calculate_upper_n_boundary(stop);

        int number_of_pythagorean_triplets = 0;

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

            for (int m = lower_m_boundary + world_rank;
                     m < upper_m_boundary;
                     m += 2 * world_size)
            {
                number_of_pythagorean_triplets += (int)(gcd(m, n) == 1);
            }
        }

        results[i] = number_of_pythagorean_triplets;
    }

    #ifdef HAVE_MPI
    MPI_Reduce(world_rank ? results : MPI_IN_PLACE,
               results, number_of_input_sets,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    #endif

    if (world_rank == 0)
    {
        for (int i = 0; i < number_of_input_sets; ++i)
        {
            printf("%d\n", results[i]);
        }
    }
    
    #ifdef HAVE_MPI
    MPI_Finalize();
    #endif
    
    free(input_sets);
    free(results);

    return EXIT_SUCCESS;
}

