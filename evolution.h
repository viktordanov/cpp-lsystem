#ifndef EVOLUTION_H
#define EVOLUTION_H


#include "distributions.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>
#include <execution>
#include <omp.h>

#include "evol_operators.h"
#include "mem_pool.h"


template<typename GeneType>
using FitnessFunction = std::function<double(const Individual<GeneType> *)>;


// Configuration class
template<typename GeneType>
struct Configuration {
    int population_size;
    int generations;
    float initial_mutation_rate;
    float exp_dropoff_rate;
    std::vector<GeneDefinition<GeneType>> gene_definitions;
    std::shared_ptr<SelectionStrategy<GeneType>> selection_strategy;
    std::shared_ptr<CrossoverStrategy<GeneType>> crossover_strategy;
    std::shared_ptr<MutationStrategy<GeneType>> mutation_strategy;

    FitnessFunction<GeneType> fitness_function;
};


struct RunStatistics {
    std::vector<double> best_fitness_history;
    int generations_to_converge;
    int plateau_length;
};

// Evolutionary Algorithm class
template<typename GeneType>
class EvolutionaryAlgorithm {
private:
    Configuration<GeneType> config;
    std::shared_ptr<ProbabilityDistribution> uniform_dist;
    MemoryPool<Individual<GeneType>> individual_pool;

public:
    explicit EvolutionaryAlgorithm(const Configuration<GeneType> &configuration)
            : config(configuration),
              uniform_dist(std::make_shared<PresampledDistribution>(new UniformDistribution(), 10000000)),
              individual_pool(configuration.population_size * 2) {
    }

    void set_population_size(int size) {
        config.population_size = size;
    }

    void set_generations(int generation) {
        config.generations = generation;
    }

    Individual<GeneType> run() {
        std::vector<Individual<GeneType> *> population = initialize_population();
        auto avg_time_per_gen = 0.0;

        for (int gen = 0; gen < config.generations; ++gen) {
            auto now = std::chrono::high_resolution_clock::now();
            evaluate_fitness(population);

            if (gen % 10 == 0) {
                print_generation_info(gen, population, avg_time_per_gen);
            }

            std::vector<Individual<GeneType> *> new_population;
            new_population.reserve(config.population_size);
            float adj_mut_rate = config.initial_mutation_rate * std::pow(
                    config.exp_dropoff_rate, gen / static_cast<float>(config.generations));

#pragma omp parallel
            {
                std::vector<Individual<GeneType> *> local_population;
                local_population.reserve(config.population_size / omp_get_num_threads());

#pragma omp for nowait
                for (int i = 0; i < config.population_size; ++i) {
                    Individual<GeneType> *parent1 = config.selection_strategy->select(population);
                    Individual<GeneType> *parent2 = config.selection_strategy->select(population);
                    Individual<GeneType> *child = individual_pool.construct(config.gene_definitions);
                    config.crossover_strategy->crossover(parent1, parent2, child);
                    config.mutation_strategy->mutate(child, config.gene_definitions, adj_mut_rate);
                    local_population.push_back(child);
                }

#pragma omp critical
                {
                    new_population.insert(new_population.end(), local_population.begin(), local_population.end());
                }
            }

            // Destroy old individuals
            for (auto individual: population) {
                individual_pool.destroy(individual);
            }

            population = std::move(new_population);

            auto end = std::chrono::high_resolution_clock::now();
            avg_time_per_gen = (avg_time_per_gen * gen + std::chrono::duration_cast<
                    std::chrono::milliseconds>(end - now).count()) / (gen + 1);
        }

        evaluate_fitness(population);
        Individual<GeneType> *best_individual = *std::min_element(population.begin(), population.end(),
                                                                  [](const Individual<GeneType> *a,
                                                                     const Individual<GeneType> *b) {
                                                                      return a->fitness < b->fitness;
                                                                  });

        Individual<GeneType> result = *best_individual;

        // Clean up remaining individuals
        for (auto individual: population) {
            individual_pool.destroy(individual);
        }

        return result;
    }


    std::pair<Individual<GeneType>, RunStatistics> run_with_stats() {
        std::vector<Individual<GeneType> *> population = initialize_population();
        auto avg_time_per_gen = 0.0;
        RunStatistics stats;

        for (int gen = 0; gen < config.generations; ++gen) {
            // use omp time to measure time
            auto start = omp_get_wtime();
            evaluate_fitness(population);

            std::vector<Individual<GeneType> *> new_population;
            new_population.reserve(config.population_size);
            float adj_mut_rate = config.initial_mutation_rate * std::pow(
                    0.25, gen / static_cast<float>(config.generations));

#pragma omp parallel
            {
                std::vector<Individual<GeneType> *> local_population;
                local_population.reserve(config.population_size / omp_get_num_threads());

#pragma omp for nowait
                for (int i = 0; i < config.population_size; ++i) {
                    Individual<GeneType> *parent1 = config.selection_strategy->select(population);
                    Individual<GeneType> *parent2 = config.selection_strategy->select(population);
                    Individual<GeneType> *child = individual_pool.construct(config.gene_definitions);
                    config.crossover_strategy->crossover(parent1, parent2, child);
                    config.mutation_strategy->mutate(child, config.gene_definitions, adj_mut_rate);
                    local_population.push_back(child);
                }

#pragma omp critical
                {
                    new_population.insert(new_population.end(), local_population.begin(), local_population.end());
                }
            }

            // Destroy old individuals
            for (auto individual: population) {
                individual_pool.destroy(individual);
            }

            population = std::move(new_population);

            auto end = omp_get_wtime();
            avg_time_per_gen = (avg_time_per_gen * gen + (end - start) * 1000) / (gen + 1);
        }

        evaluate_fitness(population);
        Individual<GeneType> *best_individual = *std::min_element(population.begin(), population.end(),
                                                                  [](const Individual<GeneType> *a,
                                                                     const Individual<GeneType> *b) {
                                                                      return a->fitness < b->fitness;
                                                                  });

        stats.best_fitness_history.reserve(config.generations);
        for (auto individual: population) {
            stats.best_fitness_history.push_back(individual->fitness);
        }

        stats.generations_to_converge = std::distance(stats.best_fitness_history.begin(),
                                                      std::min_element(stats.best_fitness_history.begin(),
                                                                       stats.best_fitness_history.end()));

        stats.plateau_length = 0;
        for (int i = stats.generations_to_converge; i < stats.best_fitness_history.size(); ++i) {
            if (stats.best_fitness_history[i] == stats.best_fitness_history[stats.generations_to_converge]) {
                stats.plateau_length++;
            } else {
                break;
            }
        }

        Individual<GeneType> result = *best_individual;

        // Clean up remaining individuals
        for (auto individual: population) {
            individual_pool.destroy(individual);
        }


        return {result, stats};

    }

private:
    std::vector<Individual<GeneType> *> initialize_population() {
        // resize the pool to fit the population size
        individual_pool.resize(config.population_size * 2);

        std::vector<Individual<GeneType> *> population;
        population.reserve(config.population_size);
        for (int i = 0; i < config.population_size; ++i) {
            Individual<GeneType> *individual = individual_pool.construct(config.gene_definitions);
            population.push_back(individual);
        }
        return population;
    }

    void evaluate_fitness(std::vector<Individual<GeneType> *> &population) {
#pragma omp parallel for
        for (size_t i = 0; i < population.size(); ++i) {
            population[i]->fitness = config.fitness_function(population[i]);
        }
    }

    void
    print_generation_info(int gen, const std::vector<Individual<GeneType> *> &population, double avg_time_per_gen) {
        std::cout << "Generation " << gen + 1 << " / " << config.generations << std::endl;
        auto *best_individual = *std::min_element(population.begin(), population.end(),
                                                  [](const Individual<GeneType> *a, const Individual<GeneType> *b) {
                                                      return a->fitness < b->fitness;
                                                  });
        std::cout << "Best fitness: " << best_individual->fitness << std::endl;
        std::cout << "Average time per generation: " << avg_time_per_gen << "ms" << std::endl;
    }
};

template<typename GeneType>
std::unique_ptr<EvolutionaryAlgorithm<GeneType>> create_evolutionary_algorithm(const Configuration<GeneType> &config) {
    return std::make_unique<EvolutionaryAlgorithm<GeneType>>(config);
}

#endif // EVOLUTION_H

