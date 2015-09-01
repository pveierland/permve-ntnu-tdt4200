#include <chrono>
#include <cmath>
#include <string>

static const char LogTable256[256] = 
{
    #define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
};

inline unsigned do_lg(unsigned v)
{
    unsigned r;     // r will be lg(v)
    register unsigned int t, tt; // temporaries
    
    if (tt = v >> 16)
    {
          r = (t = tt >> 8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
    }
    else 
    {
          r = (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v];
    }
    return r;
}

int
main(const int argc, const char* argv[])
{
    const int start      = std::atoi(argv[1]);
    const int stop       = std::atoi(argv[2]);
    const int intervals  = std::atoi(argv[3]);
    const int iterations = std::atoi(argv[4]);
    const int steps      = (stop - start) / intervals;

    std::printf("#N\tstd::log(n) [ns]\n");

    for (int n = start; n <= stop; n += steps)
    {
        const auto t1 = std::chrono::system_clock::now();

        for (int i = 0; i != iterations; ++i)
        {
            volatile auto x = std::log(n);
            //volatile auto x = do_lg(n);//(static_cast<double>(n));
        }

        const auto t2 = std::chrono::system_clock::now();

        std::printf("%d %f\n", n, static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) / iterations);
    }
}

