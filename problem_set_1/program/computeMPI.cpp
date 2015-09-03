// NTNU TDT4200 Fall 2015 Problem Set 1: MPI Intro
// permve@stud.ntnu.no

#include <boost/mpi.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace
{
    namespace config
    {
        const int master_node = 0;
        const int result_tag  = 0;
    }
}

double
sum_inverse_log_skip_multiples_of_two(
    const int node_index, const int node_count, const int start, const int stop)
{
    double sum = 0.0;

    int end_point = stop;

    double log_of_two = std::log2(2.0);
    double log_of_e   = std::log2(std::exp(1.0));

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
                sum += log_of_e / std::log2(static_cast<double>(x));
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
                    const auto l = std::log2(static_cast<double>(x));
                    sum += log_of_e / l + log_of_e / (log_of_two + l) + log_of_e / std::log2(static_cast<double>(2 * x + 1));
                }

                if (is_midpoint_odd)
                {
                    sum += log_of_e / std::log2(static_cast<double>(2 * node_end)) + log_of_e / std::log2(static_cast<double>(2 * node_end + 1));
                }
            }
            else
            {
                for (int x = node_start; x != node_end; ++x)
                {
                    const auto l = std::log2(static_cast<double>(x));
                    sum += log_of_e / l + log_of_e / (log_of_two + l) + log_of_e / std::log2(static_cast<double>(2 * x + 1));
                }
            }

            if (end_point & 1 && node_index == 0)
            {
                sum += log_of_e / std::log2(static_cast<double>(end_point - 1));
            }

            end_point = start_point;
        }
    } 
}

std::pair<int, int>
read_user_input(const int argc, const char* argv[])
{
    if (argc != 3)
    {
        throw std::runtime_error(
            "This program requires two parameters:\n"
            "the start and end specifying a range of positive integers "
            "in which start is 2 or greater, and end is greater than start.\n");
    }

    const int start = std::stoi(argv[1]);
    const int stop  = std::stoi(argv[2]);

    if (start < 2 || stop <= start)
    {
        throw std::runtime_error(
            "Start must be greater than 2 and the end must be larger than start.\n");
	}

    return std::make_pair(start, stop);
}

int
main(const int argc, const char* argv[])
{
    boost::mpi::environment  env{};
    boost::mpi::communicator world{};

    const bool is_master = (world.rank() == config::master_node);

    try
    {
        const auto start_stop = read_user_input(argc, argv);
        
        const auto t1 = std::chrono::system_clock::now();

        const auto this_node_sum = sum_inverse_log_skip_multiples_of_two(
            world.rank(), world.size(), start_stop.first, start_stop.second);
    
        if (is_master)
        {
            // Master node: gather and sum calculations for all nodes
            // Do not care about order of results; only the number of results
    
            double total_node_sum = this_node_sum;
    
            for (int i = 1; i != world.size(); ++i)
            {
                double received_node_sum;
                world.recv(boost::mpi::any_source, config::result_tag, received_node_sum);
                total_node_sum += received_node_sum;
            }
            
            const auto t2 = std::chrono::system_clock::now();

            const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            
            std::printf("%ld %f\n", nanoseconds, total_node_sum);
        }
        else
        {
            // Slave node: send calculation to master
            world.send(config::master_node, config::result_tag, this_node_sum);
        }
    }
    catch (const std::exception& error)
    {
        if (is_master)
        {
            std::cerr << error.what();
        }
        return 1;
    }
    
    return 0;
}

