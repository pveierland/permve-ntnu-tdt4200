#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <functional>

double
do_sum_sieve(const int node_index, const int node_count, const int start, const int stop)
{
    int end_point = stop;

    double two = std::log2(2);
    double e   = std::log2(std::exp(1));
    double sum = 0.0;

    while (true)
    {
        // Divide mid-point and start-point by 2:
        int mid_point = end_point >> 1;
        int start_point = mid_point >> 1;

        bool is_midpoint_odd = mid_point & 1;

        mid_point &= ~1;

        if (start_point < start || ((mid_point - start_point) < 2 * node_count))
        {
            const int per_node = (end_point - start) / node_count;

            int node_start = start + per_node * node_index;
            int node_end = node_start + per_node;

            if (node_index == node_count - 1)
            {
                node_end = end_point;
            }
                
            // Half-open range:
            for (int x = node_start; x != node_end; ++x)
            {
                sum += e / std::log2(x);
            }
            
            return sum;
        }
        else
        {
            const int per_node = (mid_point - start_point) / node_count;

            int node_start = start_point + per_node * node_index;
            int node_end = node_start + per_node;
            
            if (node_index == node_count - 1)
            {
                node_end = mid_point;
            
                for (int x = node_start; x != node_end; ++x)
                {
                    const auto l = std::log2(x);
                    sum += e / l + e / (two + l) + e / std::log2(2 * x + 1);
                }

                if (is_midpoint_odd)
                {
                    sum += e / std::log2(2 * node_end) + e / std::log2(2 * node_end + 1);
                }
            }
            else
            {
                for (int x = node_start; x != node_end; ++x)
                {
                    const auto l = std::log2(x);
                    sum += e / l + e / (two + l) + e / std::log2(2 * x + 1);
                }
            }

            if (end_point & 1 && node_index == 0)
            {
                sum += e / std::log2(end_point - 1);
            }

            end_point = start_point;
        }
    } 
}

inline
double
do_sum_brute(const int start, const int stop)
{
    double sum = 0.0;

    for (int n = start; n != stop; ++n)
    {
        sum += 1.0 / std::log(n);
    }

    return sum;
}

inline
double
do_sum_brute2(const int start, const int stop)
{
    double sum = 0.0;

    double e = std::log2(std::exp(1));

    for (int n = start; n != stop; ++n)
    {
        sum += e / std::log2(n);
    }

    return sum;
}

inline
void
benchmark(std::function<double(int, int)> f, int start, int stop)
{
    const auto t1 = std::chrono::system_clock::now();

    const auto sum = f(start, stop);

    std::printf("%f\n", sum);

    const auto t2 = std::chrono::system_clock::now();

    std::printf("microseconds = %ld\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
}

int
main(const int argc, const char* argv[])
{
    const int start      = std::atoi(argv[1]);
    const int stop       = std::atoi(argv[2]);

    benchmark(std::bind(do_sum_sieve, 0, 1, std::placeholders::_1, std::placeholders::_2), start, stop);
    benchmark(do_sum_brute, start, stop);
    benchmark(do_sum_brute2, start, stop);
}

