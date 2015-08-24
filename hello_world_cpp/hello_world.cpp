#include <boost/format.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

#include <iostream>
#include <sstream>

int main()
{
    boost::mpi::environment env{};
    boost::mpi::communicator world{};

    const auto message = boost::str(boost::format("Greetings from process %d of %d") % (world.rank() + 1) % world.size());

    if (world.rank() != 0)
    {
        world.send(0, 0, message);
    }
    else
    {
        std::cout << message << std::endl;

        for (decltype(world.size()) i = 1; i != world.size(); ++i)
        {
            std::string received{};
            world.recv(i, 0, received);
            std::cout << received << std::endl;
        }
    }

    return 0;
}

