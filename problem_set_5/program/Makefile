serial: pythagoreanTriplets.c
	gcc -std=c99 -Wall -Wextra -pedantic -o serial_program pythagoreanTriplets.c -lm

omp: pythagoreanTriplets.c
	gcc -std=c99 -Wall -Wextra -pedantic -o omp_program pythagoreanTriplets.c -lm -fopenmp -DHAVE_OPENMP

mpi: pythagoreanTriplets.c
	mpicc -std=c99 -Wall -Wextra -pedantic -o mpi_program pythagoreanTriplets.c -lm -DHAVE_MPI

combined: pythagoreanTriplets.c
	mpicc -std=c99 -Wall -Wextra -pedantic -o combined_program pythagoreanTriplets.c -lm -fopenmp -DHAVE_MPI -DHAVE_OPENMP

all: serial omp mpi combined

clean:
	rm -f *program
