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

    double sum = 0.0;//stop - start - 1;

    for (int n = start; n < stop; ++n)
    {
        sum += std::log(std::exp(1)) / std::log(n);
    }

    std::printf("%f\n", sum);
}

