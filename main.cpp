#include <cassert>
#include <map>
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include "types.h"
#include "distributions.h"
#include "lsystem.h"
#include "parse.h"

void test_rule_parsing();


int main(int argc, char const* argv[])
{
    test_rule_parsing();

    std::vector<Token> test_axiom = {"C", "A"};

    UniformDistribution uniform_dist_0{};
    PresampledDistribution uniform_dist(&uniform_dist_0, 300000);
    KumaraswamyDistribution kumaraswamy_dist(5, 1.5);

    LSystem test_lsystem(test_axiom, {
                             {"A", "1 *C:p_0[&0,12,0.2,1,6] A A"},
                         }, &uniform_dist);
    test_lsystem.set_dist(0, &kumaraswamy_dist);


    test_lsystem.print_current_state(std::cout);
    for (int i = 0; i < 12 * 3; i++)
    {
        test_lsystem.iterate(1);
        test_lsystem.print_current_state(std::cout);
    }

    auto now = std::chrono::system_clock::now();
    int runs = 1000000;
    for (int i = 0; i < runs; i++)
    {
        test_lsystem.reset();
        test_lsystem.iterate(50);
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - now;
    std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_seconds).count() /
        runs << "ns\n";

    std::vector<Token> axiom = {
        "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
        "A", "A", "A", "A", "A", "A"
    };

    KumaraswamyDistribution dist(1.5, 5);
    PresampledDistribution presampled_dist(&dist, 300000);

    LSystem lsystem(axiom, {
                        {
                            "A",
                            {
                                "1 B; 1 C; 1 D; 1 E; 1 F; 1 G; 1 H; 1 I; 1 J; 1 K; 1 L; 1 M; 1 N; 1 O; 1 P; 1 Q; 1 R; 1 S; 1 T; 1 U; 1 V; 1 W; 1 X; 1 Y; 1 Z"
                            }
                        }
                    }, &presampled_dist);

    const auto start = std::chrono::system_clock::now();

    for (int i = 0; i < 9; i++)
    {
        // lsystem.reset();
        lsystem.iterate(1);
        lsystem.print_current_state(std::cout);
    }
    end = std::chrono::system_clock::now();

    // take last state for analysis (take it from ostream)
    // make stream to outputto
    // count distribution of tokens
    std::map<TokenStateId, int> token_count;

    for (int i = 0; i < 1000; i++)
    {
        lsystem.reset();
        lsystem.iterate(1);
        std::vector<TokenStateId> ids = lsystem.current_state_bytes();
        for (auto& token : ids)
        {
            token_count[token]++;
        }
    }

    for (auto& [token, count] : token_count)
    {
        std::cout << int(token) << ": " << count << std::endl;
    }

    elapsed_seconds = end - start;
    std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_seconds).
        count() << "ns\n";

    // performance test
    runs = 1000000;
    const int max_iters = 1;

    const auto start_perf = std::chrono::system_clock::now();
    for (int i = 0; i < runs; i++)
    {
        lsystem.reset();
        lsystem.iterate(max_iters);
    }
    const auto end_perf = std::chrono::system_clock::now();
    const auto elapsed_perf = end_perf - start_perf;
    std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_perf / runs).
        count() << "ns\n";


    return 0;
}

void test_rule_parsing()
{
    std::vector<std::tuple<std::string, std::string, std::vector<WeightedRule>>> test_cases = {
        // Simple rule without catalyst
        {"A", "1 B", {WeightedRule{1, CatalystPosition::None, "", FixedActivationProbability{1}, {"B"}}}},
        // Simple rule with catalyst
        {"B", "1 *C:0.5 A", {WeightedRule{1, CatalystPosition::Left, "C", FixedActivationProbability{0.5f}, {"A"}}}},
        // Rule with multiple products
        {"C", "1 A B C", {WeightedRule{1, CatalystPosition::None, "", FixedActivationProbability{1}, {"A", "B", "C"}}}},
        {
            "C_1", "1 *A B C",
            {WeightedRule{1, CatalystPosition::Left, "A", FixedActivationProbability{1.f}, {"B", "C"}}}
        },
        // Rule with fixed activation probability
        {
            "D", "1:0.5 D E F",
            {WeightedRule{1, CatalystPosition::None, "", FixedActivationProbability{0.5f}, {"D", "E", "F"}}}
        },
        // Rule with named activation probability
        {
            "E", "1 E*:p_1[E,10,0,0,0] E F",
            {
                WeightedRule{
                    1, CatalystPosition::Right, "E", NamedActivationProbability{1, Token("E"), 10.f, 0, 0, 0},
                    {"E", "F"}
                }
            }
        },
        // Rule with catalyst and named activation probability
        {"F", "1 *F:0.6 G", {WeightedRule{1, CatalystPosition::Left, "F", FixedActivationProbability{0.6f}, {"G"}}}},
        // Rule with catalyst and named activation probability with argument
        {
            "G", "1 *G:p_2[&0,13,0,1,1] G H",
            {
                WeightedRule{
                    1, CatalystPosition::Left, "G",
                    NamedActivationProbability{2, GlobalMetaHeuristic::Length, 13, 0, 1, 1}, {"G", "H"}
                }
            }
        },
        // Rule with catalyst and named activation probability with token argument
        {
            "H", "1 *H:p_3[&0,1.3,0.2,0.8,2.1] H I",
            {
                WeightedRule{
                    1, CatalystPosition::Left, "H",
                    NamedActivationProbability{3, GlobalMetaHeuristic::Length, 1.3, 0.2, 0.8, 2.1}, {"H", "I"}
                }
            }
        },
    };

    for (auto& [predecessor, rule_str, expected_rules] : test_cases)
    {
        std::cout << "Testing rule: " << rule_str << std::endl;
        std::istringstream linestream(rule_str);
        std::vector<WeightedRule> rules = parse_rule(rule_str);
        assert(rules.size() == expected_rules.size());
        for (int i = 0; i < rules.size(); i++)
        {
            assert(rules[i].weight == expected_rules[i].weight);
            assert(rules[i].catalyst_position == expected_rules[i].catalyst_position);
            assert(rules[i].catalyst == expected_rules[i].catalyst);
            if (std::holds_alternative<FixedActivationProbability>(rules[i].activation_probability))
            {
                assert(std::holds_alternative<FixedActivationProbability>(expected_rules[i].activation_probability));
                assert(
                    std::get<FixedActivationProbability>(rules[i].activation_probability).value == std::get<
                    FixedActivationProbability>(expected_rules[i].activation_probability).value);
            }
            else
            {
                assert(std::holds_alternative<NamedActivationProbability>(expected_rules[i].activation_probability));
                NamedActivationProbability named_activation = std::get<NamedActivationProbability>(
                    rules[i].activation_probability);
                NamedActivationProbability expected_named_activation = std::get<NamedActivationProbability>(
                    expected_rules[i].activation_probability);
                assert(named_activation.distribution == expected_named_activation.distribution);
                assert(named_activation.meta_heuristic == expected_named_activation.meta_heuristic);
                for (int j = 0; j < 4; j++)
                {
                    assert(
                        std::abs(named_activation.probability_shape_constants[j] - expected_named_activation.
                            probability_shape_constants[j]) < 1e-6);
                }
            }
            for (int j = 0; j < rules[i].products.size(); j++)
            {
                assert(rules[i].products[j] == expected_rules[i].products[j]);
            }
        }
    }

    std::vector<std::string> invalid_rule_strs = {
        // Rule with invalid activation probability
        "1 A B C:0.5",
        // Rule with invalid catalyst
        "1 *:0.5 A",
    };

    for (auto& rule_str : invalid_rule_strs)
    {
        try
        {
            parse_rule(rule_str);
            std::cout << "Invalid rule string did not throw exception: " << rule_str << std::endl;
            assert(false);
        }
        catch (const std::invalid_argument& e)
        {
            continue;
        }
    }


    std::cout << "All test cases passed successfully." << std::endl;
}
