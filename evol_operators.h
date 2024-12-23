#ifndef EVOL_OPERATORS_H
#define EVOL_OPERATORS_H

#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include "distributions.h"

// Gene definition structure
template<typename T>
struct GeneDefinition {
    T min_value;
    T max_value;
    T mutation_sigma;
};

// Individual class
template<typename GeneType>
class Individual {
public:
    std::vector<GeneType> genes;
    double fitness;

    Individual() : fitness(std::numeric_limits<double>::max()) {}

    explicit Individual(const std::vector<GeneDefinition<GeneType>>& gene_defs)
            : genes(gene_defs.size()), fitness(std::numeric_limits<double>::max()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        for (size_t i = 0; i < gene_defs.size(); ++i) {
            std::uniform_real_distribution<> dis(gene_defs[i].min_value, gene_defs[i].max_value);
            genes[i] = static_cast<GeneType>(dis(gen));
        }
    }
};

// Selection strategy interface
template<typename GeneType>
class SelectionStrategy {
public:
    virtual Individual<GeneType>* select(const std::vector<Individual<GeneType>*>& population) const = 0;
    virtual ~SelectionStrategy() = default;
};

// Crossover strategy interface
template<typename GeneType>
class CrossoverStrategy {
public:
    virtual void crossover(const Individual<GeneType>* parent1, const Individual<GeneType>* parent2, Individual<GeneType>* child) const = 0;
    virtual ~CrossoverStrategy() = default;
};

// Mutation strategy interface
template<typename GeneType>
class MutationStrategy {
public:
    virtual void mutate(Individual<GeneType>* individual, const std::vector<GeneDefinition<GeneType>>& gene_defs, float mutation_rate) = 0;
    virtual ~MutationStrategy() = default;
};

// Tournament selection implementation
template<typename GeneType>
class TournamentSelection : public SelectionStrategy<GeneType> {
private:
    int tournament_size;

public:
    explicit TournamentSelection(int size) : tournament_size(size) {}

    Individual<GeneType>* select(const std::vector<Individual<GeneType>*>& population) const override {
        if (population.empty()) {
            throw std::runtime_error("Population is empty");
        }
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, population.size() - 1);
        Individual<GeneType>* best = population[dist(rng)];
        for (int i = 1; i < tournament_size; ++i) {
            Individual<GeneType>* contender = population[dist(rng)];
            if (contender->fitness < best->fitness) {
                best = contender;
            }
        }
        return best;
    }
};

// Single-point crossover implementation
template<typename GeneType>
class SinglePointCrossover : public CrossoverStrategy<GeneType> {
private:

public:
    explicit SinglePointCrossover() = default;

    void crossover(const Individual<GeneType>* parent1, const Individual<GeneType>* parent2, Individual<GeneType>* child) const override {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, parent1->genes.size() - 1);
        size_t mid_point = dist(rng);
        for (size_t i = 0; i < mid_point; ++i) {
            child->genes[i] = parent1->genes[i];
        }
        for (size_t i = mid_point; i < parent1->genes.size(); ++i) {
            child->genes[i] = parent2->genes[i];
        }
    }
};

// Gaussian mutation implementation
template<typename GeneType>
class GaussianMutation : public MutationStrategy<GeneType> {
private:
    std::mt19937 rng;

public:
    explicit GaussianMutation()
            :  rng(std::random_device{}()) {}

    void mutate(Individual<GeneType>* individual, const std::vector<GeneDefinition<GeneType>>& gene_defs, float mutation_rate) override {
        thread_local std::uniform_real_distribution<float> uniform_dist(0, 1);
        for (size_t i = 0; i < individual->genes.size(); ++i) {
            if (uniform_dist(rng) < mutation_rate) {
                std::normal_distribution<float> normal_dist(0, gene_defs[i].mutation_sigma);
                individual->genes[i] += static_cast<GeneType>(normal_dist(rng));
                individual->genes[i] = std::clamp(individual->genes[i], gene_defs[i].min_value, gene_defs[i].max_value);
            }
        }
    }
};

// Cauchy mutation implementation
template<typename GeneType>
class CauchyMutation : public MutationStrategy<GeneType> {
private:
    std::mt19937 rng;

public:
    explicit CauchyMutation()
            :  rng(std::random_device{}()) {}

    void mutate(Individual<GeneType>* individual, const std::vector<GeneDefinition<GeneType>>& gene_defs, float mutation_rate) override {
        thread_local std::uniform_real_distribution<float> uniform_dist(0, 1);
        for (size_t i = 0; i < individual->genes.size(); ++i) {
            if (uniform_dist(rng) < mutation_rate) {
                std::cauchy_distribution<float> cauchy_dist(0, gene_defs[i].mutation_sigma);
                individual->genes[i] += static_cast<GeneType>(cauchy_dist(rng));
                individual->genes[i] = std::clamp(individual->genes[i], gene_defs[i].min_value, gene_defs[i].max_value);
            }
        }
    }
};

// Roulette Wheel Selection implementation
template<typename GeneType>
class RouletteWheelSelection : public SelectionStrategy<GeneType> {
public:
    Individual<GeneType>* select(const std::vector<Individual<GeneType>*>& population) const override {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double total_fitness = std::accumulate(population.begin(), population.end(), 0.0,
                                               [](double sum, const Individual<GeneType>* ind) {
                                                   return sum + 1.0 / ind->fitness;
                                               });
        double random_point = dist(rng) * total_fitness;
        double current_sum = 0.0;
        for (const auto* individual : population) {
            current_sum += 1.0 / individual->fitness;
            if (current_sum >= random_point) {
                return const_cast<Individual<GeneType>*>(individual);
            }
        }
        return const_cast<Individual<GeneType>*>(population.back());
    }
};

// Rank-based Selection implementation
template<typename GeneType>
class RankBasedSelection : public SelectionStrategy<GeneType> {
public:
    Individual<GeneType>* select(const std::vector<Individual<GeneType>*>& population) const override {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<Individual<GeneType>*> sorted_population = population;
        std::sort(sorted_population.begin(), sorted_population.end(),
                  [](const Individual<GeneType>* a, const Individual<GeneType>* b) { return a->fitness < b->fitness; });
        double total_rank = population.size() * (population.size() + 1) / 2.0;
        double random_point = dist(rng) * total_rank;
        double current_sum = 0.0;
        for (size_t i = 0; i < sorted_population.size(); ++i) {
            current_sum += i + 1;
            if (current_sum >= random_point) {
                return sorted_population[i];
            }
        }
        return sorted_population.back();
    }
};

// Uniform Crossover implementation
template<typename GeneType>
class UniformCrossover : public CrossoverStrategy<GeneType> {
private:

public:
    explicit UniformCrossover() = default;

    void crossover(const Individual<GeneType>* parent1, const Individual<GeneType>* parent2, Individual<GeneType>* child) const override {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < parent1->genes.size(); ++i) {
            child->genes[i] = (dist(rng) < 0.5) ? parent1->genes[i] : parent2->genes[i];
        }
    }
};

// Arithmetic Crossover implementation
template<typename GeneType>
class ArithmeticCrossover : public CrossoverStrategy<GeneType> {
private:
    const std::vector<GeneDefinition<GeneType>>& gene_defs;

public:
    explicit ArithmeticCrossover(const std::vector<GeneDefinition<GeneType>>& defs) : gene_defs(defs) {}

    void crossover(const Individual<GeneType>* parent1, const Individual<GeneType>* parent2, Individual<GeneType>* child) const override {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        GeneType alpha = static_cast<GeneType>(dist(rng));
        for (size_t i = 0; i < parent1->genes.size(); ++i) {
            child->genes[i] = alpha * parent1->genes[i] + (1 - alpha) * parent2->genes[i];
            child->genes[i] = std::clamp(child->genes[i], gene_defs[i].min_value, gene_defs[i].max_value);
        }
    }
};

// Uniform Mutation implementation
template<typename GeneType>
class UniformMutation : public MutationStrategy<GeneType> {
private:

public:

    explicit UniformMutation(){};

    void mutate(Individual<GeneType>* individual, std::vector<GeneDefinition<GeneType>>& gene_defs, float mutation_rate) override {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < individual->genes.size(); ++i) {
            if (dist(rng) < mutation_rate) {
                individual->genes[i] = static_cast<GeneType>(dist(rng) * (gene_defs[i].max_value - gene_defs[i].min_value) + gene_defs[i].min_value);
            }
        }
    }
};

// Polynomial Mutation implementation
template<typename GeneType>
class PolynomialMutation : public MutationStrategy<GeneType> {
private:
    GeneType eta_m;

public:
    PolynomialMutation(GeneType eta_m = 20)
            : eta_m(eta_m) {}

    void mutate(Individual<GeneType>* individual, const std::vector<GeneDefinition<GeneType>>& gene_defs, float mutation_rate) override {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < individual->genes.size(); ++i) {
            if (dist(rng) < mutation_rate) {
                GeneType delta1 = (individual->genes[i] - gene_defs[i].min_value) / (gene_defs[i].max_value - gene_defs[i].min_value);
                GeneType delta2 = (gene_defs[i].max_value - individual->genes[i]) / (gene_defs[i].max_value - gene_defs[i].min_value);
                GeneType rnd = static_cast<GeneType>(dist(rng));
                GeneType mut_pow = 1.0 / (eta_m + 1.0);
                GeneType deltaq;
                if (rnd <= 0.5) {
                    GeneType xy = 1.0 - delta1;
                    GeneType val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (std::pow(xy, (eta_m + 1.0)));
                    deltaq = std::pow(val, mut_pow) - 1.0;
                } else {
                    GeneType xy = 1.0 - delta2;
                    GeneType val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (std::pow(xy, (eta_m + 1.0)));
                    deltaq = 1.0 - std::pow(val, mut_pow);
                }
                individual->genes[i] += deltaq * (gene_defs[i].max_value - gene_defs[i].min_value);
                individual->genes[i] = std::clamp(individual->genes[i], gene_defs[i].min_value, gene_defs[i].max_value);
            }
        }
    }
};

#endif // EVOL_OPERATORS_H
