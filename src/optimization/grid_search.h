#ifndef GRID_SEARCH_H
#define GRID_SEARCH_H

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <memory>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include "optimization/evolution.h"

template <typename GeneType>
class GridSearch {
public:
    struct NamedValue {
        std::string name;
        std::variant<int, float, std::vector<GeneDefinition<GeneType>>, std::shared_ptr<void>> value;
    };

    using ConfigUpdater = std::function<void(Configuration<GeneType>&, const NamedValue&)>;

    struct ParameterSpace {
        std::map<std::string, std::vector<NamedValue>> parameters;
        std::map<std::string, ConfigUpdater> updaters;


    public:
        template<typename T, typename F>
        void register_parameter(const std::string& param_name,
                                const std::vector<std::pair<std::string, T>>& named_values,
                                F updater) {
            std::vector<NamedValue> param_values;
            for (const auto& [name, value] : named_values) {
                param_values.push_back({name, value});
            }
            parameters[param_name] = param_values;
            updaters[param_name] = [updater](Configuration<GeneType>& config, const NamedValue& named_value) {
                updater(config, std::get<T>(named_value.value));
            };
        }
    };

    struct Result {
        std::map<std::string, NamedValue> config;
        Individual<GeneType> best_individual;
        RunStatistics stats;
        double execution_time;

        std::string get_run_id() const {
            std::string run_id;
            for (const auto& [param_name, named_value] : config) {
                run_id += named_value.name + "_";
            }
            return run_id;
        }
    };

private:
    ParameterSpace param_space;
    FitnessFunction<GeneType> fitness_function;
    std::vector<Result> results;

    void generate_configurations(std::vector<std::map<std::string, NamedValue>>& configs,
                                 std::map<std::string, NamedValue>& current_config,
                                 size_t depth) {
        if (depth == param_space.parameters.size()) {
            configs.push_back(current_config);
            return;
        }
        auto it = std::next(param_space.parameters.begin(), depth);
        for (const auto& named_value : it->second) {
            current_config[it->first] = named_value;
            generate_configurations(configs, current_config, depth + 1);
        }
    }

    Result run_configuration(const std::map<std::string, NamedValue>& config) {
        Configuration<GeneType> ea_config;
        ea_config.fitness_function = fitness_function;
        for (const auto& [key, named_value] : config) {
            if (param_space.updaters.count(key) > 0) {
                param_space.updaters.at(key)(ea_config, named_value);
            }
        }
        auto ea = create_evolutionary_algorithm(ea_config);
        auto run_start = std::chrono::high_resolution_clock::now();
        auto [best_individual, stats] = ea->run_with_stats();
        auto run_end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double>(run_end - run_start).count();

        return {config, best_individual, stats, execution_time};
    }

    size_t run_count = 0;

    void print_table_header() const {
        std::cout << std::left << std::setw(5) << "Run" << " | ";
        for (const auto& [param_name, _] : param_space.parameters) {
            std::cout << std::setw(20) << param_name << " | ";
        }
        std::cout << std::setw(15) << "Best Fitness" << " | "
                  << std::setw(10) << "Time (s)" << std::endl;


        std::cout << std::string(5 + 3 + 20 * param_space.parameters.size() + 15 + 10, '-') << std::endl;
    }

    void print_table_row(const Result& result) const {
        std::cout << std::left << std::setw(5) << run_count << " | ";
        for (const auto& [param_name, _] : param_space.parameters) {
            std::cout << std::setw(20) << result.config.at(param_name).name << " | ";
        }

        // now with scientific notation
        std::cout << std::setw(15) << std::scientific << result.best_individual.fitness << " | "
                  << std::setw(10) << std::defaultfloat << result.execution_time << std::endl;
        std::cout.flush(); // Ensure the output is displayed immediately
    }



public:
    GridSearch(const ParameterSpace& space, const FitnessFunction<GeneType>& fitness_func)
            : param_space(space), fitness_function(fitness_func) {}

    void run() {
        std::vector<std::map<std::string, NamedValue>> configs;
        std::map<std::string, NamedValue> current_config;
        generate_configurations(configs, current_config, 0);

        auto start_time = std::chrono::high_resolution_clock::now();

        results.resize(configs.size());

        print_table_header();

        #pragma omp parallel for shared(results)  schedule(dynamic) num_threads(2)
        for (size_t i = 0; i < configs.size(); ++i) {
            results[i] = run_configuration(configs[i]);

            #pragma omp critical
            {
                print_table_row(results[i]);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "\nTotal execution time: " << total_time << "s" << std::endl;
    }


    void summary() const {

        // find and print the best result by best fitness AND fewest generations
        auto best = std::min_element(results.begin(), results.end(), [](const Result& a, const Result& b) {
            // fit and gen
            if (a.best_individual.fitness == b.best_individual.fitness) {
                return a.stats.generations_to_converge < b.stats.generations_to_converge;
            }
            return a.best_individual.fitness < b.best_individual.fitness;
        });

        auto best_run_id = best->get_run_id();
        auto best_index = std::distance(results.begin(), best);

        std::cout << "Best result:" << std::endl;
        std::cout << "Run ID: " << best_run_id << std::endl;
        std::cout << "Fitness: " << results[best_index].best_individual.fitness << std::endl;
        std::cout << "Generations to converge: " << results[best_index].stats.generations_to_converge << std::endl;
        std::cout << "Plateau length: " << results[best_index].stats.plateau_length << std::endl;
        std::cout << "Execution time: " << results[best_index].execution_time << "s" << std::endl;
        // print the best configuration
        std::cout << "Configuration:" << std::endl;
        for (const auto& [key, named_value] : results[best_index].config) {
            std::cout << key << ": " << named_value.name << std::endl;
        }
        // print the best individual
        std::cout << "Best individual:" << std::endl;
        for (const auto& gene : results[best_index].best_individual.genes) {
            std::cout << gene << " ";
        }
    }

    void save_results(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return;
        }

        // Write header
        for (const auto& [key, _] : param_space.parameters) {
            file << key << ",";
        }
        file << "Best_Fitness,Generations_to_Converge,Plateau_Length,Execution_Time\n";

        // Write results
        for (const auto& result : results) {
            for (const auto& [key, named_value] : result.config) {
                file << named_value.name << ",";
            }
            file << result.best_individual.fitness << ","
                 << result.stats.generations_to_converge << ","
                 << result.stats.plateau_length << ","
                 << result.execution_time << "\n";
        }
        file.close();
    }

    const std::vector<Result>& get_results() const {
        return results;
    }

    const Result& get_best_result() const {
        auto best = std::min_element(results.begin(), results.end(), [](const Result& a, const Result& b) {
            // fit and gen
            if (a.best_individual.fitness == b.best_individual.fitness) {
                return a.stats.generations_to_converge < b.stats.generations_to_converge;
            }
            return a.best_individual.fitness < b.best_individual.fitness;
        });
        return *best;
    }

    std::unique_ptr<EvolutionaryAlgorithm<GeneType>> get_best_ea() const {
        auto best = get_best_result();
        Configuration<GeneType> ea_config;
        ea_config.fitness_function = fitness_function;
        for (const auto& [key, named_value] : best.config) {
            if (param_space.updaters.count(key) > 0) {
                param_space.updaters.at(key)(ea_config, named_value);
            }
        }
        return create_evolutionary_algorithm(ea_config);
    }
};

template <typename GeneType>
std::unique_ptr<GridSearch<GeneType>> create_grid_search(
        const typename GridSearch<GeneType>::ParameterSpace& space,
        const FitnessFunction<GeneType>& fitness_func) {
    return std::make_unique<GridSearch<GeneType>>(space, fitness_func);
}

#endif // GRID_SEARCH_H
