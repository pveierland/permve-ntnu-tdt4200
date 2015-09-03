#include <chrono>
#include <cmath>
#include <string>

int
main(const int argc, const char* argv[])
{
    const int start = std::atoi(argv[1]);
    const int stop  = std::atoi(argv[2]);

    const auto t1 = std::chrono::system_clock::now();
    
    double sum = 0.0;

    for (int i = start; i != stop; ++i)
    {
        sum += 1.0 / std::log(static_cast<double>(i));
    }

    const auto t2 = std::chrono::system_clock::now();

    const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    std::printf("%ld %f\n", nanoseconds, sum);
}

