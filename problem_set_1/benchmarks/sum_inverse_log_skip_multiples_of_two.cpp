#include <chrono>
#include <cmath>
#include <string>

double
sum_inverse_log_skip_multiples_of_two(
    const int node_index, const int node_count, const int start, const int stop)
{
    double sum = 0.0;

    int end_point = stop;

    double log_of_two = std::log2(2);
    double log_of_e   = std::log2(std::exp(1));

    while (true)
    {
        // Divide mid-point and start-point by 2:
        int mid_point = end_point >> 1;
        int start_point = mid_point >> 1;

        bool is_midpoint_odd = mid_point & 1;

        mid_point &= ~1;

        if (start_point < start || ((mid_point - start_point) < 2 * node_count))
        {
            // Not possible to split remaining data. Revert to plain iteration.
            const int per_node = (end_point - start) / node_count;

            int node_start = start + per_node * node_index;
            int node_end = node_start + per_node;

            if (node_index == node_count - 1)
            {
                node_end = end_point;
            }
                
            for (int x = node_start; x != node_end; ++x)
            {
                sum += log_of_e / std::log2(x);
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
                    sum += log_of_e / l + log_of_e / (log_of_two + l) + log_of_e / std::log2(2 * x + 1);
                }

                if (is_midpoint_odd)
                {
                    sum += log_of_e / std::log2(2 * node_end) + log_of_e / std::log2(2 * node_end + 1);
                }
            }
            else
            {
                for (int x = node_start; x != node_end; ++x)
                {
                    const auto l = std::log2(x);
                    sum += log_of_e / l + log_of_e / (log_of_two + l) + log_of_e / std::log2(2 * x + 1);
                }
            }

            if (end_point & 1 && node_index == 0)
            {
                sum += log_of_e / std::log2(end_point - 1);
            }

            end_point = start_point;
        }
    } 
}

int
main(const int argc, const char* argv[])
{
    const int start = std::atoi(argv[1]);
    const int stop  = std::atoi(argv[2]);

    const auto t1 = std::chrono::system_clock::now();
    const auto sum = sum_inverse_log_skip_multiples_of_two(0, 1, start, stop);
    const auto t2 = std::chrono::system_clock::now();

    const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    std::printf("%ld %f\n", nanoseconds, sum);
}

