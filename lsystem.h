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

    TokenStateId empty_state_id;
    std::map<Token, TokenStateId> token_bytes;

    Token bytes_token[255];
    ByteProductionRule** byte_rules;

    TokenStateId param_bytes[255];
    int params[128];

    std::vector<TokenStateId> current_state;
    std::vector<TokenStateId> next_state;

    ProbabilityDistribution* uniform_dist;

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
