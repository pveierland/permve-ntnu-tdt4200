serial: pythagoreanTriplets.c
	gcc -std=c99 -Wall -o serial_program pythagoreanTriplets.c -lm

omp: pythagoreanTriplets.c
	gcc -std=c99 -Wall -o omp_program pythagoreanTriplets.c -lm -fopenmp -DHAVE_OPENMP

mpi: pythagoreanTriplets.c
	mpicc -std=c99 -Wall -o mpi_program pythagoreanTriplets.c -lm -DHAVE_MPI

combined: pythagoreanTriplets.c
	mpicc -std=c99 -Wall -o combined_program pythagoreanTriplets.c -lm -fopenmp -DHAVE_MPI -DHAVE_OPENMP

all: serial omp mpi combined

clean:
	rm -f *program
