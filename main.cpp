#include <cassert>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cmath>


#include "types.h"
#include "distributions.h"
#include "evolution.h"
#include "lsystem.h"
#include "parse.h"
#include "grid_search.h"
#include "kl_divergence.h"

void test_rule_parsing() {
    std::vector<std::tuple<std::string, std::string, std::vector<WeightedRule> > > test_cases = {
        // Simple rule without catalyst
        {"A", "1 B", {WeightedRule{1, CatalystPosition::None, "", FixedActivationProbability{1}, {"B"}}}},
        // Simple rule with catalyst
        {
            "B", "1 *C:0.5 A", {
                WeightedRule{
                    1, CatalystPosition::Left, "C", FixedActivationProbability{0.5f},
                    {"A"}
                }
            }
        },
        // Rule with multiple products
        {
            "C", "1 A B C", {
                WeightedRule{
                    1, CatalystPosition::None, "", FixedActivationProbability{1},
                    {"A", "B", "C"}
                }
            }
        },
        {
            "C_1", "1 *A B C",
            {
                WeightedRule{
                    1, CatalystPosition::Left, "A", FixedActivationProbability{1.f},
                    {"B", "C"}
                }
            }
        },
        // Rule with fixed activation probability
        {
            "D", "1:0.5 D E F",
            {
                WeightedRule{
                    1, CatalystPosition::None, "", FixedActivationProbability{0.5f},
                    {"D", "E", "F"}
                }
            }
        },
        // Rule with named activation probability
        {
            "E", "1 E*:p_1[E,10,0,0,0] E F",
            {
                WeightedRule{
                    1, CatalystPosition::Right, "E",
                    NamedActivationProbability{1, Token("E"), 10.f, 0, 0, 0},
                    {"E", "F"}
                }
            }
        },
        // Rule with catalyst and named activation probability
        {
            "F", "1 *F:0.6 G", {
                WeightedRule{
                    1, CatalystPosition::Left, "F", FixedActivationProbability{0.6f},
                    {"G"}
                }
            }
        },
        // Rule with catalyst and named activation probability with argument
        {
            "G", "1 *G:p_2[&0,13,0,1,1] G H",
            {
                WeightedRule{
                    1, CatalystPosition::Left, "G",
                    NamedActivationProbability{2, GlobalMetaHeuristic::Length, 13, 0, 1, 1},
                    {"G", "H"}
                }
            }
        },
        // Rule with catalyst and named activation probability with token argument
        {
            "H", "1 *H:p_3[&0,1.3,0.2,0.8,2.1] H I",
            {
                WeightedRule{
                    1, CatalystPosition::Left, "H",
                    NamedActivationProbability{
                        3, GlobalMetaHeuristic::Length, 1.3, 0.2, 0.8,
                        2.1
                    },
                    {"H", "I"}
                }
            }
        },
    };

    for (auto &[predecessor, rule_str, expected_rules]: test_cases) {
        std::cout << "Testing rule: " << rule_str << std::endl;
        std::istringstream linestream(rule_str);
        std::vector<WeightedRule> rules = parse_rule(rule_str);
        assert(rules.size() == expected_rules.size());
        for (int i = 0; i < rules.size(); i++) {
            assert(rules[i].weight == expected_rules[i].weight);
            assert(rules[i].catalyst_position == expected_rules[i].catalyst_position);
            assert(rules[i].catalyst == expected_rules[i].catalyst);
            if (std::holds_alternative<FixedActivationProbability>(rules[i].activation_probability)) {
                assert(std::holds_alternative<FixedActivationProbability>(expected_rules[i].activation_probability));
                assert(
                    std::get<FixedActivationProbability>(rules[i].activation_probability).value == std::get<
                    FixedActivationProbability>(expected_rules[i].activation_probability).value);
            } else {
                assert(std::holds_alternative<NamedActivationProbability>(expected_rules[i].activation_probability));
                NamedActivationProbability named_activation = std::get<NamedActivationProbability>(
                    rules[i].activation_probability);
                NamedActivationProbability expected_named_activation = std::get<NamedActivationProbability>(
                    expected_rules[i].activation_probability);
                assert(named_activation.distribution == expected_named_activation.distribution);
                assert(named_activation.meta_heuristic == expected_named_activation.meta_heuristic);
                for (int j = 0; j < 4; j++) {
                    assert(
                        std::abs(named_activation.probability_shape_constants[j] - expected_named_activation.
                            probability_shape_constants[j]) < 1e-6);
                }
            }
            for (int j = 0; j < rules[i].products.size(); j++) {
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

    for (auto &rule_str: invalid_rule_strs) {
        try {
            parse_rule(rule_str);
            std::cout << "Invalid rule string did not throw exception: " << rule_str << std::endl;
            assert(false);
        } catch (const std::invalid_argument &e) {
            continue;
        }
    }


    std::cout << "All test cases passed successfully." << std::endl;
}


LSystem create_lsystem_from_individual(const Individual<float> &individual) {
    auto *uniform_dist = new UniformDistribution(0, 1);
    auto *kumaraswamy_dist = new KumaraswamyDistribution(individual.genes[0], individual.genes[1]);

    const std::vector<Token> axiom = {"l", "L"};
    const TokenSet variables = {"L"};
    const TokenSet constants = {"l", ""};

    const std::map<Token, ProductionRule> rules = {
        {
            "L", {
                "L",
                {
                    WeightedRule{
                        1.0f,
                        CatalystPosition::Left,
                        "l",
                        NamedActivationProbability{
                            0,
                            GlobalMetaHeuristic::Length,
                            {
                                individual.genes[2],
                                individual.genes[3],
                                individual.genes[4],
                                individual.genes[5]
                            }
                        },
                        {"L", "L"}
                    }
                }
            }
        }
    };

    LSystem l_system(axiom, variables, constants, rules, uniform_dist);
    l_system.set_dist(0, kumaraswamy_dist);

    return l_system;
}

using TargetWithPriority = std::tuple<int, Distribution, float>;

void grid_search_antenna();
void grid_search_schwefel_func();

int main(int argc, char const *argv[]) {
    omp_set_num_threads(8);
    omp_set_nested(1);

//     std::vector<GeneDefinition<float> > gene_defs = {
//         {0.0f, 50.0f, 0.5f}, // kumaraswamy_alpha
//         {0.0f, 50.0f, 0.5f}, // kumaraswamy_beta
//         {0.0f, 120.0f, 1.0f}, // probability_shape_constant_1
//         {0.0f, 0.9f, 0.01f}, // probability_shape_constant_2
//         {0.2f, 1.0f, 0.01f}, // probability_shape_constant_3
//         {0.0f, 20.0f, 0.1f} // probability_shape_constant_4
//     };
//     ////    // Define target distributions at various checkpoints
//     std::vector<TargetWithPriority> targets = {
//         {10, create_target_distribution(10), 1.0},
//         {20, create_target_distribution(20), 1.0},
//     };
//
//
//     auto fitness_function = [&gene_defs, &targets](const Individual<float> *individual) {
//         constexpr int NUM_RUNS = 100;
//         double total_fitness = 0.0;
//
// #pragma omp parallel reduction(+:total_fitness) num_threads(2)
//         for (int run = 0; run < NUM_RUNS; ++run) {
//             double run_divergence = 0.0;
//             LSystem l_system = create_lsystem_from_individual(*individual);
//             int previous_checkpoint = 0;
//
//             for (const auto &target: targets) {
//                 int checkpoint = std::get<0>(target);
//                 Distribution target_dist = std::get<1>(target);
//                 double priority = std::get<2>(target);
//
//                 int iterations = checkpoint - previous_checkpoint;
//                 l_system.iterate(iterations);
//
//                 Distribution actual_dist = get_distribution(l_system);
//                 double divergence = kl_divergence(target_dist, actual_dist);
//
//                 run_divergence += divergence * priority;
//                 previous_checkpoint = checkpoint;
//             }
//
//             total_fitness += run_divergence;
//         }
//
//         total_fitness /= NUM_RUNS;
//
//         return total_fitness;
//     };
//
//
//     GridSearch<float>::ParameterSpace param_space;
//
//     param_space.register_parameter<int>("population_size",
//                                         {{"100", 100}},
//                                         [](Configuration<float> &config, const int &value) {
//                                             config.population_size = value;
//                                         }
//     );
//
//     param_space.register_parameter<float>("mutation_rate",
//                                           {
//                                               {"0.9", 0.9f},
//                                               {"0.5", 0.5f}
//                                           },
//                                           [](Configuration<float> &config, const float &value) {
//                                               config.initial_mutation_rate = value;
//                                           }
//     );
//
//     param_space.register_parameter<int>("generation_count",
//                                         {{"100", 100}},
//                                         [](Configuration<float> &config, const int &value) {
//                                             config.generations = value;
//                                         }
//     );
//
//     param_space.register_parameter<std::vector<GeneDefinition<float> > >("gene_definitions",
//                                                                          {
//                                                                              {"LSystem Parameters", gene_defs}
//                                                                          },
//                                                                          [](Configuration<float> &config,
//                                                                             const std::vector<GeneDefinition<float> > &
//                                                                             value) {
//                                                                              config.gene_definitions = value;
//                                                                          }
//     );
//
//     param_space.register_parameter<std::shared_ptr<void> >("selection_strategy",
//                                                            {
//                                                                {
//                                                                    "tour 50",
//                                                                    std::make_shared<TournamentSelection<float> >(50)
//                                                                },
//                                                            },
//                                                            [](Configuration<float> &config,
//                                                               const std::shared_ptr<void> &value) {
//                                                                config.selection_strategy = std::static_pointer_cast<
//                                                                    SelectionStrategy<float> >(
//                                                                    value);
//                                                            }
//     );
//
//     param_space.register_parameter<std::shared_ptr<void> >("crossover_strategy",
//                                                            {
//                                                                {
//                                                                    "single",
//                                                                    std::make_shared<SinglePointCrossover<float> >()
//                                                                },
//                                                            },
//                                                            [](Configuration<float> &config,
//                                                               const std::shared_ptr<void> &value) {
//                                                                config.crossover_strategy = std::static_pointer_cast<
//                                                                    CrossoverStrategy<float> >(
//                                                                    value);
//                                                            }
//     );
//
//     param_space.register_parameter<std::shared_ptr<void> >("mutation_strategy",
//                                                            {
//                                                                {
//                                                                    "gaussian",
//                                                                    std::make_shared<GaussianMutation<float> >()
//                                                                },
//                                                            },
//                                                            [](Configuration<float> &config,
//                                                               const std::shared_ptr<void> &value) {
//                                                                config.mutation_strategy = std::static_pointer_cast<
//                                                                    MutationStrategy<float> >(
//                                                                    value);
//                                                            }
//     );
//
//     auto grid_search = create_grid_search<float>(param_space, fitness_function);
//     grid_search->run();
//     grid_search->summary();
//
//     auto ea = grid_search->get_best_ea();
//     ea->set_population_size(5000);
//     ea->set_generations(100);
//     auto best = ea->run();
//
//     std::cout << "Best individual:\n";
//     for (int i = 0; i < best.genes.size(); i++) {
//         std::cout << "Gene " << i << ": " << best.genes[i] << "\n";
//     }
//     std::cout << "Fitness: " << -fitness_function(&best) << "\n";
//
//
//     LSystem l_system = create_lsystem_from_individual(best);
//     // show each iter and  the length of the current state with the current state
//     for (int i = 0; i < 100; i++) {
//         l_system.iterate(1);
//         std::cout << "Iteration " << i << ": " << l_system.current_state_bytes().size() << " tokens ";
//         l_system.print_current_state(std::cout);
//     }
//

    // grid_search_antenna();
    grid_search_schwefel_func();
    return 0;
}


void grid_search_antenna() {
    const float C = 3e8;
    const float TARGET_FREQUENCY = 6e9;
    const float WAVELENGTH = C / TARGET_FREQUENCY;

    auto fitness_function = [&WAVELENGTH, &TARGET_FREQUENCY, &C](const Individual<float> *individual) {
        float L = individual->genes[0];
        float W = individual->genes[1];
        float h = individual->genes[2];
        float εr = individual->genes[3];

        // Calculate the actual resonant frequency
        float f = C / (2 * L * std::sqrt(εr));

        // Calculate bandwidth (simplified model)
        float BW = 3.77 * (h / W) * (εr - 1) / std::sqrt(εr);

        // Calculate directivity (simplified model)
        float D = 4 * M_PI * L * W / (WAVELENGTH * WAVELENGTH);

        // Calculate efficiency (simplified model)
        float η = 1 - (1 / (1 + 10 * h / WAVELENGTH));


        float frequency_error = std::abs(f - TARGET_FREQUENCY);
        float fitness = D * η * BW - frequency_error;

        return -fitness;
    };

    GridSearch<float>::ParameterSpace param_space;

    param_space.register_parameter<int>("population_size",
                                        {{"100", 100}},
                                        [](Configuration<float> &config, const int &value) {
                                            config.population_size = value;
                                        }
    );

    // Register mutation rate
    param_space.register_parameter<float>("mutation_rate",
                                          {
                                              {"0.01", 0.01f},
                                              {"0.1", 0.1f},
                                              {"0.5", 0.5f}
                                          },
                                          [](Configuration<float> &config, const float &value) {
                                              config.initial_mutation_rate = value;
                                          }
    );

    // Register generation count
    param_space.register_parameter<int>("generation_count",
                                        {{"100", 100}},
                                        [](Configuration<float> &config, const int &value) {
                                            config.generations = value;
                                        }
    );

    // Register gene definitions for antenna parameters
    param_space.register_parameter<std::vector<GeneDefinition<float> > >("gene_definitions",
                                                                         {
                                                                             {
                                                                                 "Antenna Parameters",
                                                                                 {
                                                                                     GeneDefinition<float>{
                                                                                         0.5f * WAVELENGTH,
                                                                                         2.0f * WAVELENGTH,
                                                                                         0.001f
                                                                                     }, // L
                                                                                     GeneDefinition<float>{
                                                                                         0.5f * WAVELENGTH,
                                                                                         2.0f * WAVELENGTH,
                                                                                         0.001f
                                                                                     }, // W
                                                                                     GeneDefinition<float>{
                                                                                         0.01f * WAVELENGTH,
                                                                                         0.1f * WAVELENGTH,
                                                                                         0.0001f
                                                                                     }, // h
                                                                                     GeneDefinition<float>{
                                                                                         1.0f,
                                                                                         12.0f,
                                                                                         0.1f
                                                                                     }, // εr
                                                                                 }
                                                                             }
                                                                         },
                                                                         [](Configuration<float> &config,
                                                                            const std::vector<GeneDefinition<float> > &
                                                                            value) {
                                                                             config.gene_definitions = value;
                                                                         }
    );

    // Register selection strategies
    param_space.register_parameter<std::shared_ptr<void> >("selection_strategy",
                                                           {
                                                               {
                                                                   "tour 10",
                                                                   std::make_shared<TournamentSelection<float> >(
                                                                       10)
                                                               },
                                                               {
                                                                   "tour 50",
                                                                   std::make_shared<TournamentSelection<float> >(
                                                                       50)
                                                               },
                                                               {
                                                                   "roulette",
                                                                   std::make_shared<RouletteWheelSelection<float> >()
                                                               }
                                                           },
                                                           [](Configuration<float> &config,
                                                              const std::shared_ptr<void> &value) {
                                                               config.selection_strategy = std::static_pointer_cast<
                                                                   SelectionStrategy<float> >(
                                                                   value);
                                                           }
    );

    // Register crossover strategy
    param_space.register_parameter<std::shared_ptr<void> >("crossover_strategy",
                                                           {
                                                               {
                                                                   "uniform",
                                                                   std::make_shared<UniformCrossover<float> >()
                                                               },
                                                               {
                                                                   "single-point",
                                                                   std::make_shared<SinglePointCrossover<float>>()
                                                               },
                                                           },
                                                           [](Configuration<float> &config,
                                                              const std::shared_ptr<void> &value) {
                                                               config.crossover_strategy = std::static_pointer_cast<
                                                                   CrossoverStrategy<float> >(
                                                                   value);
                                                           }
    );

    // Register mutation strategy
    param_space.register_parameter<std::shared_ptr<void> >("mutation_strategy",
                                                           {
                                                               {
                                                                   "gaussian",
                                                                   std::make_shared<GaussianMutation<float> >()
                                                               },
                                                           },
                                                           [](Configuration<float> &config,
                                                              const std::shared_ptr<void> &value) {
                                                               config.mutation_strategy = std::static_pointer_cast<
                                                                   MutationStrategy<float> >(
                                                                   value);
                                                           }
    );

    // Create the GridSearch object
    auto grid_search = create_grid_search<float>(param_space, fitness_function);

    // Run the grid search
    grid_search->run();
    grid_search->summary();

    // Evolve selected configuration
    auto ea = grid_search->get_best_ea();
    ea->set_generations(200);
    auto best = ea->run();

    // Output results
    std::cout << "Best antenna design:\n";
    std::cout << "L: " << best.genes[0] << " m\n";
    std::cout << "W: " << best.genes[1] << " m\n";
    std::cout << "h: " << best.genes[2] << " m\n";
    std::cout << "εr: " << best.genes[3] << "\n";
    std::cout << "Fitness: " << -fitness_function(&best) << "\n";

    // Calculate and display the resonant frequency
    float f = C / (2 * best.genes[0] * std::sqrt(best.genes[3]));
    std::cout << "Resonant Frequency: " << f / 1e9 << " GHz\n";


    // Save results
    grid_search->save_results("grid_search_results.csv");
}


void grid_search_schwefel_func() {
    // Schwefel function
    auto fitness_function = [](const Individual<float> *individual) {
        float sum = 0.0;
        for (size_t i = 0; i < individual->genes.size(); ++i) {
            sum += individual->genes[i] * std::sin(std::sqrt(std::abs(individual->genes[i])));
        }
        return 418.9829 * individual->genes.size() - sum;
    };

    GridSearch<float>::ParameterSpace param_space;

    // Register population size
    param_space.register_parameter<int>("population_size",
                                        {{"50", 50}, {"100", 100}, {"200", 200}},
                                        [](Configuration<float> &config, const int &value) {
                                            config.population_size = value;
                                        }
    );

    // Register mutation rate
    param_space.register_parameter<float>("mutation_rate",
                                           {{"0.01", 0.01}, {"0.05", 0.05}, {"0.1", 0.1}},
                                           [](Configuration<float> &config, const float &value) {
                                               config.initial_mutation_rate = value;
                                           }
    );

    // Register generation count
    param_space.register_parameter<int>("generation_count",
                                        {{"100", 100},{"1000",1000}},
                                        [](Configuration<float> &config, const int &value) {
                                            config.generations = value;
                                        }
    );

    // Register gene definitions for Schwefel function
    // Typically, the search space for Schwefel function is [-500, 500] for each dimension
    param_space.register_parameter<std::vector<GeneDefinition<float> > >("gene_definitions",
                                                                          {
                                                                              {
                                                                                  "Schwefel Parameters",
                                                                                  {
                                                                                      GeneDefinition<float>{
                                                                                          -500.0, 500.0, 0.1
                                                                                      },
                                                                                  }
                                                                              }
                                                                          },
                                                                          [](Configuration<float> &config,
                                                                             const std::vector<GeneDefinition<float> >
                                                                             &value) {
                                                                              config.gene_definitions = value;
                                                                          }
    );

    // Register selection strategies
    param_space.register_parameter<std::shared_ptr<void> >("selection_strategy",
                                                           {
                                                               {
                                                                   "tour 5",
                                                                   std::make_shared<TournamentSelection<float> >(5)
                                                               },
                                                               {
                                                                   "tour 10",
                                                                   std::make_shared<TournamentSelection<float> >(10)
                                                               },
                                                               {
                                                                   "roulette",
                                                                   std::make_shared<RouletteWheelSelection<float> >()
                                                               }
                                                           },
                                                           [](Configuration<float> &config,
                                                              const std::shared_ptr<void> &value) {
                                                               config.selection_strategy = std::static_pointer_cast<
                                                                   SelectionStrategy<float> >(value);
                                                           }
    );

    // Register crossover strategy
    param_space.register_parameter<std::shared_ptr<void> >("crossover_strategy",
                                                           {
                                                               {
                                                                   "uniform",
                                                                   std::make_shared<UniformCrossover<float> >()
                                                               },
                                                               {
                                                                   "single-point",
                                                                   std::make_shared<SinglePointCrossover<float> >()
                                                               }
                                                           },
                                                           [](Configuration<float> &config,
                                                              const std::shared_ptr<void> &value) {
                                                               config.crossover_strategy = std::static_pointer_cast<
                                                                   CrossoverStrategy<float> >(value);
                                                           }
    );

    // Register mutation strategy
    param_space.register_parameter<std::shared_ptr<void> >("mutation_strategy",
                                                           {
                                                               {
                                                                   "gaussian",
                                                                   std::make_shared<GaussianMutation<float> >()
                                                               },
                                                           },
                                                           [](Configuration<float> &config,
                                                              const std::shared_ptr<void> &value) {
                                                               config.mutation_strategy = std::static_pointer_cast<
                                                                   MutationStrategy<float> >(value);
                                                           }
    );

    // Create the GridSearch object
    auto grid_search = create_grid_search<float>(param_space, fitness_function);

    // Run the grid search
    grid_search->run();
    grid_search->summary();

    // Evolve selected configuration
    auto ea = grid_search->get_best_ea();
    ea->set_generations(1000); // Run for more generations with the best configuration
    auto best = ea->run();

    // Output results
    std::cout << "Best solution found:\n";
    for (size_t i = 0; i < best.genes.size(); ++i) {
        std::cout << "x" << i + 1 << ": " << best.genes[i] << "\n";
    }
    std::cout << "Fitness: " << fitness_function(&best) << "\n";

    // Save results
    grid_search->save_results("schwefel_grid_search_results.csv");
}
