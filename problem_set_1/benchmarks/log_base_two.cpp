#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

int
main(const int argc, const char* argv[])
{
    const int start      = std::atoi(argv[1]);
    const int stop       = std::atoi(argv[2]);
    const int intervals  = std::atoi(argv[3]);
    const int iterations = std::atoi(argv[4]);
    const int steps      = (stop - start) / intervals;

    std::cout << "#N\tCost per call [ns]" << std::endl;

    for (int n = start; n <= stop; n += steps)
    {
        const auto t1 = std::chrono::system_clock::now();

        for (int i = 0; i != iterations; ++i)
        {
            volatile auto x = std::log2(static_cast<double>(n));
        }

        const auto t2 = std::chrono::system_clock::now();

        std::printf("%d %f\n", n, static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) / iterations);
    }
}

