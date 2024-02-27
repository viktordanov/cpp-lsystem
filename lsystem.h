//
// Created by vikimaster2 on 2/18/24.
//

#ifndef LSYSTEM_H
#define LSYSTEM_H

#include <vector>
#include <map>
#include <string>
#include <iostream>

#include "types.h"
#include "distributions.h"

class LSystem
{
public:
    std::vector<Token> axiom;
    TokenSet variables;
    TokenSet constants;
    std::map<Token, ProductionRule> rules;

    std::map<Token, TokenStateId> token_bytes;

    TokenStateId empty_state_id;
    std::array<Token, 255> bytes_token;
    std::array<ByteProductionRule*, 255> byte_rules;

    std::array<TokenStateId, 255> param_bytes;
    std::array<int, 128> params;

    std::vector<TokenStateId> current_state;
    std::vector<TokenStateId> next_state;

    ProbabilityDistribution* uniform_dist;
    std::array<ProbabilityDistribution*, 4> dists;
    void set_dist(const int, ProbabilityDistribution*);

    void reset();

    void encode_tokens();
    void iterate(int);
    void apply_rules(int);
    void apply_rules_once(const std::vector<TokenStateId>& input, std::vector<TokenStateId>& output);

    LSystem(std::vector<Token> axiom, const std::map<Token, std::string>& rules, ProbabilityDistribution* dist);

    [[nodiscard]] std::vector<TokenStateId> current_state_bytes() const
    {
        return this->current_state;
    }

    void print_current_state(std::ostream& os) const
    {
        for (auto& token : this->current_state)
        {
            os << this->bytes_token[token];
        }
        os << std::endl;
    }
};


#endif //LSYSTEM_H
