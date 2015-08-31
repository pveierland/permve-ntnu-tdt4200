#include <iostream>
#include <utility>

#include <gtest/gtest.h>
using namespace ::testing;

/**
 * Given a node index, node count and half-open range described
 * by a start and stop index; calculate the start and stop index
 * relative to a given node.
 *
 * @param node_index Zero-based node index [0..inf)
 * @param node_count Total number of nodes [1..inf)
 * @param start      Start of range (inclusive)
 * @param stop       End of range (exclusive)
 *
 * @return Pair with first (inclusive) and last (exclusive) indexes
 *         relative to specified node.
 *
 * @pre node_index >= 0
 * @pre node_count > node_index
 * @pre start >= stop
 */
std::pair<int, int>
calculate_node_indexes(const int node_index,
                       const int node_count,
                       const int start,
                       const int stop)
{
    const int total_calculations            = stop - start;
    const int minimum_calculations_per_node = total_calculations / node_count;

    const int remaining_calculations =
        total_calculations - node_count * minimum_calculations_per_node;

    const int this_node_start =
        start + 
        minimum_calculations_per_node * node_index +
        std::min(remaining_calculations, node_index);

    const int this_node_calculations =
        minimum_calculations_per_node +
        ((node_index + 1 <= remaining_calculations) ? 1 : 0);

    return std::make_pair(this_node_start, this_node_start + this_node_calculations);
}

void
validate_node_indexes(const int                               node_count,
                      const int                               start,
                      const int                               stop,
                      const std::vector<std::pair<int, int>>& indexes)
{
    ASSERT_GE(stop, start);

    ASSERT_EQ(start, indexes[0].first);
    ASSERT_GE(indexes[0].second, indexes[0].first);

    int total = indexes[0].second - indexes[0].first;

    for (std::size_t i = 1; i != indexes.size(); ++i)
    {
        ASSERT_EQ(indexes[i - 1].second, indexes[i].first);
        ASSERT_GE(indexes[i].second, indexes[i].first);
        total += indexes[i].second - indexes[i].first;
    }

    ASSERT_EQ(indexes.rbegin()->second, stop);
    ASSERT_EQ(stop - start, total);
}

TEST(calculate_node_indexes, less_work_than_nodes)
{
    const std::vector<std::pair<int, int>> indexes
    {
        calculate_node_indexes(0, 5, 10, 12),
        calculate_node_indexes(1, 5, 10, 12),
        calculate_node_indexes(2, 5, 10, 12),
        calculate_node_indexes(3, 5, 10, 12),
        calculate_node_indexes(4, 5, 10, 12)
    };

    validate_node_indexes(5, 10, 12, indexes);
}

TEST(calculate_node_indexes, no_work)
{
    const std::vector<std::pair<int, int>> indexes
    {
        calculate_node_indexes(0, 5, 0, 0),
        calculate_node_indexes(1, 5, 0, 0),
        calculate_node_indexes(2, 5, 0, 0),
        calculate_node_indexes(3, 5, 0, 0),
        calculate_node_indexes(4, 5, 0, 0)
    };

    validate_node_indexes(5, 0, 0, indexes);
}

TEST(calculate_node_indexes, even_work)
{
    const int start = 0;
    const int stop  = 20;

    const std::vector<std::pair<int, int>> indexes
    {
        calculate_node_indexes(0, 5, start, stop),
        calculate_node_indexes(1, 5, start, stop),
        calculate_node_indexes(2, 5, start, stop),
        calculate_node_indexes(3, 5, start, stop),
        calculate_node_indexes(4, 5, start, stop),
    };

    validate_node_indexes(5, start, stop, indexes);
}

TEST(calculate_node_indexes, uneven_work)
{
    const int start = 0;
    const int stop  = 8;

    const std::vector<std::pair<int, int>> indexes
    {
        calculate_node_indexes(0, 3, start, stop),
        calculate_node_indexes(1, 3, start, stop),
        calculate_node_indexes(2, 3, start, stop),
    };

    validate_node_indexes(3, start, stop, indexes);
}

TEST(calculate_node_indexes, brute_force_check)
{
    std::vector<std::pair<int, int>> indexes{};

    for (int start = 0; start != 20; ++start)
    {
        for (int stop = start; stop != 20; ++stop)
        {
            for (int node_count = 1; node_count != 30; ++node_count)
            {
                indexes.clear();

                for (int node_index = 0; node_index != node_count; ++node_index)
                {
                    indexes.push_back(calculate_node_indexes(node_index, node_count, start, stop));
                }

                validate_node_indexes(node_count, start, stop, indexes);
            }
        }
    }
}

