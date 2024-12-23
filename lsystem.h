//
// Created by vikimaster2 on 2/18/24.
//

#ifndef LSYSTEM_H
#define LSYSTEM_H

#include <vector>
#include <map>
#include <string>

#include "types.h"
#include "distributions.h"

class LSystem {
public:
    TokenSet variables;
    TokenSet constants;
    std::map<Token, ProductionRule> rules;

    std::map<Token, TokenStateId> token_bytes;

    std::vector<TokenStateId> axiom;
    TokenStateId empty_state_id;
    alignas(64) std::array<Token, 256> bytes_token;
    alignas(64) std::array<ByteProductionRule *, 256> byte_rules;

    std::array<TokenStateId, 256> param_bytes;
    std::array<int, 128> params;

    int current_iteration;
    std::vector<TokenStateId> current_state;
    std::vector<TokenStateId> next_state;

    ProbabilityDistribution *uniform_dist;
    std::array<ProbabilityDistribution *, 4> dists;

    void set_dist(const int, ProbabilityDistribution *);

    void reset();

    [[nodiscard]] float calculate_activation_probability(const ActivationStrategy &strategy, int position) const;

    void encode_tokens();

    void iterate(int);

    void apply_rules(int);

    void apply_rules_once(const std::vector<TokenStateId> &input, std::vector<TokenStateId> &output);

    LSystem(std::vector<Token> axiom, const std::map<Token, std::string>& rules, ProbabilityDistribution* dist);

    LSystem(std::vector<Token> axiom, 
                TokenSet variables,
                TokenSet constants,
                std::map<Token, ProductionRule> parsed_rules,
                ProbabilityDistribution* dist);

    std::vector<Token> decode_axiom() const;

    ~LSystem() {
        std::set<ByteProductionRule *> rules;
        std::set<ByteWeightedRule *> weighted_rules;
        unsigned long long all = 0;
        for (int i = 0; i < 256; i++) {
            // find all unique rules
            if (this->byte_rules[i] == nullptr)
                continue;

            rules.insert(this->byte_rules[i]);

            if (this->byte_rules[i]->weights.empty())
                continue;

            for (auto wt: this->byte_rules[i]->weights) {
                all++;
                weighted_rules.insert(wt);
            }
        }

        // delete all unique rules
        for (auto &rule: rules) {
            delete rule;
        }
        for (auto &wt: weighted_rules) {
            delete wt;
        }
    }

    [[nodiscard]] std::vector<TokenStateId> current_state_bytes() const {
        return this->current_state;
    }

    void print_current_state(std::ostream &os) const {
        for (auto &token: this->current_state) {
            os << this->bytes_token[token];
        }
        os << std::endl;
    }
};


#endif //LSYSTEM_H
