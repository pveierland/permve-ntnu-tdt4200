#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
using namespace ::testing;

namespace
{
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
                       const int stop,
                       const int divisor,
                       const int magnitude)
{
    const int mid_point = stop / 2;

    const int big_block_count = divisor * divisor;
    const int low_point = std::max(stop / big_block_count, start);

    const int small_block_count = (big_block_count - 1) * node_count;
    const int small_block_range = (stop - low_point) / small_block_count;

    const int this_node_start = low_point + small_block_range * node_index;

    auto x = std::make_pair(this_node_start, this_node_start + small_block_range);

    printf("big_block_count = %d low_point = %d small_block_count = %d small_block_range = %d this_node_start = %d this_node_stop = %d\n\n",
            big_block_count, low_point, small_block_count, small_block_range, this_node_start, (this_node_start + small_block_range));
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

}


bool
validate_node_vector(const std::vector<int>& veritas, const int start, const int end)
{
    bool result = true;

    if (veritas.size() != end)
    {
        printf("ALPHA");
        return false;
    }

    for (int i = 0; i != start; ++i)
    {
        if (veritas[i] != 0)
        {
            printf("CHARLIE");
            return false;
        }
    }

    for (int i = start; i != end; ++i)
    {
        if (veritas[i] != 1)
        {
            result = false;
            printf("BRAVO %d %d!", i, veritas[i]);
        }
    }

    return result;
}

void
do_work(std::vector<int>& target, const int node_index, const int node_count, const int start, const int stop)
{
    int end_point = stop;

    while (true)
    {
        // Divide mid-point and start-point by 2:
        int mid_point = end_point >> 1;
        int start_point = mid_point >> 1;

        bool is_odd_wtf = mid_point & 1;

        mid_point &= ~1;

        if (start_point < start || ((mid_point - start_point) < 2 * node_count))
        {
            const int per_node = (end_point - start) / node_count;

            int n = node_index;

            //for (int n = 0; n != node_count; ++n)
            {
                int node_start = start + per_node * n;
                int node_end = node_start + per_node;

                if (n == node_count - 1)
                {
                    node_end = end_point;
                }
                
 //               printf("BASIC node %d start %d end %d\n", n, node_start, node_end);

                // Half-open range:
                for (int x = node_start; x != node_end; ++x)
                {
                    target.at(x) += 1;
                }
            }
            
            return;
        }
        else
        {
            const int per_node = (mid_point - start_point) / node_count;

  //          printf("ADVANCED start_point = %d mid_point = %d end_point = %d per_node = %d\n",
//                    start_point, mid_point, end_point, per_node);

            int n = node_index;
            //for (int n = 0; n != node_count; ++n)
            {
                int node_start = start_point + per_node * n;
                int node_end = node_start + per_node;
                
                if (n == node_count - 1)
                {
                    node_end = mid_point;
                
                    for (int x = node_start; x != node_end; ++x)
                    {
                        target.at(x) += 1;
                        target.at(2 * x) += 1;
                        target.at(2 * x + 1) += 1;
                    }

                    if (is_odd_wtf)
                    {
                        target.at(2 * node_end) += 1;
                        target.at(2 * node_end + 1) += 1;
                    }
                }
                else
                {
                    for (int x = node_start; x != node_end; ++x)
                    {
                        target.at(x) += 1;
                        target.at(2 * x) += 1;
                        target.at(2 * x + 1) += 1;
                    }
                }

 //               printf("node %d start %d end %d\n", n, node_start, node_end);
            }

            if (end_point & 1 && n == 0)
            {
                target.at(end_point - 1) += 1;
//                printf("SPEAILSCH %d\n", end_point);
            }

            end_point = start_point;
        }
    }
}

    
TEST(FOO, BAR)
{
    std::vector<int> veritas{};

    //do_work(veritas, 0, 3, start, stop);

    for (int start = 2; start != 10; ++start)
    {
        for (int stop = start; stop != 20; ++stop)
        {
            for (int node_count = 1; node_count != 5; ++node_count)
            {
                veritas.clear();
                veritas.resize(stop);
                
  //              printf("\n\n");
  //              printf("%d %d %d\n", start, stop, node_count);

                for (int node_index = 0; node_index != node_count; ++node_index)
                {
                    do_work(veritas, node_index, node_count, start, stop);
                }

                ASSERT_TRUE(validate_node_vector(veritas, start, stop));
            }
        }
    }
}

//TEST(FOO, BAR)
//{
//    const int start = 0;
//    const int end = 100;
//    const int nodes = 3;
//
//    int end_point = end;
//
//    int it = 0;
//            
//    int node_start;
//    int node_end;
//
//    while (true)
//    {
//        const int mid_point   = end_point / 2;
//        const int start_point = mid_point / 2;
//
//        if (start_point < start || ((mid_point - start_point) < nodes))
//        {
//            const int per_node = (end_point - start) / nodes;
//
//            for (int n = 0; n != nodes; ++n)
//            {
//                node_start = start + per_node * n;
//                node_end = node_start + per_node;
//
//                if (n == nodes - 1)
//                {
//                    node_end = end_point;
//                }
//
//                printf("node %d start %d end %d\n", n, node_start, node_end);
//            }
//            
//            return;
//        }
//        else
//        {
//            const int per_node = (mid_point - start_point) / nodes;
//
//            printf("start_point = %d mid_point = %d end_point = %d per_node = %d\n",
//                    start_point, mid_point, end_point, per_node);
//
//            for (int n = 0; n != nodes; ++n)
//            {
//                node_start = start_point + per_node * n;
//                node_end = node_start + per_node;
//                
//                if (n == nodes - 1)
//                {
//                    node_end = mid_point;
//                }
//                
//                printf("node %d start %d end %d\n", n, node_start, node_end);
//            }
//
//            if (end_point & 1 && it != 0)
//            {
//                printf("SPEAILSCH %d\n", end_point);
//            }
//
//            end_point = start_point;
//
//            ++it;
//        }
//    }
////    auto a = calculate_node_indexes(0, 2, 44, 100, 2, 0);
////    auto b = calculate_node_indexes(1, 2, 44, 100, 2, 0);
////    auto c = calculate_node_indexes(2, 3, 2, 146, 2, 0);
//}

//TEST(DISABLED_calculate_node_indexes, less_work_than_nodes)
//{
//    const std::vector<std::pair<int, int>> indexes
//    {
//        calculate_node_indexes(0, 5, 10, 12),
//        calculate_node_indexes(1, 5, 10, 12),
//        calculate_node_indexes(2, 5, 10, 12),
//        calculate_node_indexes(3, 5, 10, 12),
//        calculate_node_indexes(4, 5, 10, 12)
//    };
//
//    validate_node_indexes(5, 10, 12, indexes);
//}
//
//TEST(DISABLED_calculate_node_indexes, no_work)
//{
//    const std::vector<std::pair<int, int>> indexes
//    {
//        calculate_node_indexes(0, 5, 0, 0),
//        calculate_node_indexes(1, 5, 0, 0),
//        calculate_node_indexes(2, 5, 0, 0),
//        calculate_node_indexes(3, 5, 0, 0),
//        calculate_node_indexes(4, 5, 0, 0)
//    };
//
//    validate_node_indexes(5, 0, 0, indexes);
//}
//
//TEST(DISABLED_calculate_node_indexes, even_work)
//{
//    const int start = 0;
//    const int stop  = 20;
//
//    const std::vector<std::pair<int, int>> indexes
//    {
//        calculate_node_indexes(0, 5, start, stop),
//        calculate_node_indexes(1, 5, start, stop),
//        calculate_node_indexes(2, 5, start, stop),
//        calculate_node_indexes(3, 5, start, stop),
//        calculate_node_indexes(4, 5, start, stop),
//    };
//
//    validate_node_indexes(5, start, stop, indexes);
//}
//
//TEST(DISABLED_calculate_node_indexes, uneven_work)
//{
//    const int start = 0;
//    const int stop  = 8;
//
//    const std::vector<std::pair<int, int>> indexes
//    {
//        calculate_node_indexes(0, 3, start, stop),
//        calculate_node_indexes(1, 3, start, stop),
//        calculate_node_indexes(2, 3, start, stop),
//    };
//
//    validate_node_indexes(3, start, stop, indexes);
//}
//
//TEST(DISABLED_calculate_node_indexes, brute_force_check)
//{
//    std::vector<std::pair<int, int>> indexes{};
//
//    for (int start = 0; start != 20; ++start)
//    {
//        for (int stop = start; stop != 20; ++stop)
//        {
//            for (int node_count = 1; node_count != 30; ++node_count)
//            {
//                indexes.clear();
//
//                for (int node_index = 0; node_index != node_count; ++node_index)
//                {
//                    indexes.push_back(calculate_node_indexes(node_index, node_count, start, stop));
//                }
//
//                validate_node_indexes(node_count, start, stop, indexes);
//            }
//        }
//    }
//}

