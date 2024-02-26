//
// Created by vikimaster2 on 2/18/24.
//

#ifndef TYPES_H
#define TYPES_H
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "distributions.h"

typedef std::string Token;
typedef std::set<Token> TokenSet;

// size of tokenization sets uint16
typedef uint8_t TokenSize;

typedef uint8_t TokenStateId;

inline TokenStateId make_token_state_id(const TokenSize& id, const bool& has_param)
{
    return has_param ? id | 0x80 : id;
}

inline bool has_param(const TokenStateId& id)
{
    return (id & 0x80) != 0;
}

inline TokenSize get_id(const TokenStateId& id)
{
    return id & 0x7F;
}



struct ProductionRule;
struct ByteProductionRule;

// Production rules
struct WeightedRule
{
    float weight;
    Token catalyst;
    std::vector<Token> products;
};

struct ProductionRule
{
    Token predecessor;
    std::vector<WeightedRule> weights;

    [[nodiscard]] std::vector<Token> choose_successor() const;
    void encode_tokens(ByteProductionRule* rule, std::map<Token, TokenStateId> token_bytes, ProbabilityDistribution* dist, const bool&
                       presample) const;
};

struct ByteWeightedRule
{
    float lower_limit;
    float upper_limit;
    TokenStateId catalyst;
    std::vector<TokenStateId> products;
};

class LSystem;

struct ByteProductionRule
{
    std::vector<ByteWeightedRule*> weights;
    TokenStateId predecessor;
    int currentIndex{};
    std::vector<TokenSize> preSampledWeights;

    void pre_sample(ProbabilityDistribution* distribution);
    const std::vector<TokenStateId>* choose_successor(LSystem* l, const TokenStateId& predecessor);
    std::pair<TokenSize, ByteWeightedRule*> find_rule_by_probability(float p);
};

#endif //TYPES_H
