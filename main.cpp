#include <map>
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include "types.h"
#include "distributions.h"
#include "lsystem.h"

int main(int argc, char const* argv[])
{

    std::vector<Token> axiom = { "A" };

    UniformDistribution dist{};

    LSystem lsystem(axiom, {
        {
            "A", {"1 A B C9"},
        },
            {"C7", "1 *C9 x"},
        { "B", {"1 A"} },
        { "C1", {"1 _"} },

    }, &dist);


    lsystem.print_current_state(std::cout);
    for (int i = 0; i < 10; i++)
    {
        lsystem.iterate(1);
        lsystem.print_current_state(std::cout);
    }



    // std::vector<Token> axiom = {
    //     "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
    //     "A", "A", "A", "A", "A", "A"
    // };
    //
    // KumaraswamyDistribution dist(1.5, 5);
    // PresampledDistribution presampled_dist(&dist, 300000);
    //
    // LSystem lsystem(axiom, {
    //                     {
    //                         "A",
    //                         {
    //                             "1 B; 1 C; 1 D; 1 E; 1 F; 1 G; 1 H; 1 I; 1 J; 1 K; 1 L; 1 M; 1 N; 1 O; 1 P; 1 Q; 1 R; 1 S; 1 T; 1 U; 1 V; 1 W; 1 X; 1 Y; 1 Z"
    //                         }
    //                     }
    //                 }, &presampled_dist);
    //
    // const auto start = std::chrono::system_clock::now();
    //
    // for (int i = 0; i < 9; i++)
    // {
    //     // lsystem.reset();
    //     lsystem.iterate(1);
    //     lsystem.print_current_state(std::cout);
    // }
    // const auto end = std::chrono::system_clock::now();
    //
    // // take last state for analysis (take it from ostream)
    // // make stream to outputto
    // // count distribution of tokens
    // std::map<TokenStateId, int> token_count;
    //
    // for (int i = 0; i < 1000; i++)
    // {
    //     lsystem.reset();
    //     lsystem.iterate(1);
    //     std::vector<TokenStateId> ids = lsystem.current_state_bytes();
    //     for (auto& token : ids)
    //     {
    //         token_count[token]++;
    //     }
    // }
    //
    // for (auto& [token, count] : token_count)
    // {
    //     std::cout << int(token) << ": " << count << std::endl;
    // }
    //
    // const std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_seconds).
    //     count() << "ns\n";
    //
    // // performance test
    // const int runs = 1000000;
    // const int max_iters = 1;
    //
    // const auto start_perf = std::chrono::system_clock::now();
    // for (int i = 0; i < runs; i++)
    // {
    //     lsystem.reset();
    //     lsystem.iterate(max_iters);
    // }
    // const auto end_perf = std::chrono::system_clock::now();
    // const auto elapsed_perf = end_perf - start_perf;
    // std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_perf / runs).
    //     count() << "ns\n";
    //

    return 0;
}
