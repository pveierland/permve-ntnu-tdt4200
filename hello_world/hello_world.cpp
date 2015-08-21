#include <iostream>
#include <mpi.h>
#include <sstream>

int main()
{
    int comm_sz;
    int my_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  

    std::ostringstream oss{};
    oss << "Greetings from process " << (my_rank + 1) << " of " << comm_sz << "!";

    if (my_rank != 0)
    {
        MPI_Send((void*)(oss.str().c_str()), oss.str().size() + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        std::cout << oss.str() << std::endl;
        for (int i = 1; i < comm_sz; ++i)
        {
            char buf[100] = {};
            MPI_Recv(buf, 100, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << buf << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}

