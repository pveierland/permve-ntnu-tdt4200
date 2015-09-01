// NTNU TDT4200 Fall 2015 Problem Set 1: MPI Intro
// permve@stud.ntnu.no

#include <boost/mpi.hpp>

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

/**
 * Given a node index, node count and half-open range described
 * by a start and stop index; calculate the start and stop index
 * relative to a given node.
 *
 * @param node_index       Zero-based node index [0..inf)
 * @param node_count       Total number of nodes [1..inf)
 * @param total_node_range Half-open range shared by all nodes
 *
 * @return Pair with first (inclusive) and last (exclusive) indexes
 *         relative to specified node.
 *
 * @pre node_index >= 0
 * @pre node_count > node_index
 * @pre total_node_range.second >= total_node_range.first
 */
std::pair<int, int>
calculate_node_range(const int                 node_index,
                     const int                 node_count,
                     const std::pair<int, int> total_node_range)
{
    const int total_calculations            = total_node_range.second - total_node_range.first;
    const int minimum_calculations_per_node = total_calculations / node_count;

    const int remaining_calculations =
        total_calculations - node_count * minimum_calculations_per_node;

    const int this_node_start =
        total_node_range.first +
        minimum_calculations_per_node * node_index +
        std::min(remaining_calculations, node_index);

    const int this_node_calculations =
        minimum_calculations_per_node +
        ((node_index + 1 <= remaining_calculations) ? 1 : 0);

    return std::make_pair(this_node_start,
                          this_node_start + this_node_calculations);
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
        const auto total_nodes_range = read_user_input(argc, argv);
        const auto this_node_range   = calculate_node_range(world.rank(), world.size(), total_nodes_range);
    
        double this_node_sum = 2.0;
    
        for (auto i = this_node_range.first; i != this_node_range.second; ++i)
        {
            this_node_sum -= std::log(static_cast<double>(i));
        }
    
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
    
            std::printf("The sum is: %f\n", total_node_sum);
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

