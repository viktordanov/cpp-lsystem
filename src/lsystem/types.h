//
// Created by vikimaster2 on 2/18/24.
//

#ifndef TYPES_H
#define TYPES_H
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <variant>
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

// Catalyst position indicator

enum class GlobalMetaHeuristic { Length, Position, Iteration };

enum class ActivationStrategyType { Fixed, Distribution, CubicSpline };

struct CubicSplineParams {
    std::vector<std::pair<float, float> > controlPoints;
};

struct DistributionParams {
    int distribution;
    std::variant<Token, GlobalMetaHeuristic> meta_heuristic;
    std::array<float, 4> constants;

    ~DistributionParams(){
        if (distribution == 0) {
            meta_heuristic.~variant();
        }
    }
};

struct ActivationStrategy {
    ActivationStrategyType type;

    union {
        float fixedValue{};
        DistributionParams distribution_params;
        CubicSplineParams cubicSplineParams;
    };

    ActivationStrategy() : type(ActivationStrategyType::Fixed) {}
    explicit ActivationStrategy(const float fixed_value) : type(ActivationStrategyType::Fixed), fixedValue(fixed_value) {}
    explicit ActivationStrategy(const DistributionParams& distribution_params) : type(ActivationStrategyType::Distribution), distribution_params(distribution_params) {}
    explicit ActivationStrategy(const std::vector<std::pair<float, float>>& points)
            : type(ActivationStrategyType::CubicSpline), cubicSplineParams{points} {}

    ActivationStrategy(const ActivationStrategy& other) : type(other.type) {
        if (type == ActivationStrategyType::Fixed) {
            fixedValue = other.fixedValue;
        } else {
            new(&distribution_params) DistributionParams(other.distribution_params);
        }
    }

    ~ActivationStrategy() {
        if (type == ActivationStrategyType::Distribution) {
            distribution_params.~DistributionParams();
        }
    }

    ActivationStrategy& operator=(const ActivationStrategy& other) {
        if (this == &other) {
            return *this;
        }

        if (type == ActivationStrategyType::Distribution) {
            distribution_params.~DistributionParams();
        }

        type = other.type;

        if (type == ActivationStrategyType::Fixed) {
            fixedValue = other.fixedValue;
        } else {
            new(&distribution_params) DistributionParams(other.distribution_params);
        }

        return *this;
    }

};

enum class CatalystPosition { None, Left, Right };

struct FixedActivationProbability
{
    float value;
};

struct NamedActivationProbability
{
    int distribution;
    std::variant<Token, GlobalMetaHeuristic> meta_heuristic;
    std::array<float, 4> probability_shape_constants; // normalizing constant, min, max, scale
};

typedef std::variant<FixedActivationProbability, NamedActivationProbability> ActivationProbability;


struct WeightedRule {
    float weight;
    CatalystPosition catalyst_position = CatalystPosition::None;
    Token catalyst;
    ActivationStrategy activation_strategy;
    std::vector<Token> products;
};

struct ProductionRule
{
    Token predecessor;
    std::vector<WeightedRule> weights;

    [[nodiscard]] std::vector<Token> choose_successor() const;
    void encode_tokens(ByteProductionRule* rule, std::map<Token, TokenStateId> token_bytes,
                       ProbabilityDistribution* dist, const bool&
                       presample) const;
};

struct ByteWeightedRule
{
    float lower_limit;
    float upper_limit;
    CatalystPosition catalyst_position;
    TokenStateId catalyst;
    ActivationStrategy activation_strategy;
    std::vector<TokenStateId> products;

    ~ByteWeightedRule() = default;
};

class LSystem;

struct ByteProductionRule
{
    std::vector<ByteWeightedRule*> weights;
    TokenStateId predecessor;
    int currentIndex{};
    std::vector<TokenSize> preSampledWeights;

    void pre_sample(ProbabilityDistribution* distribution);
    const std::vector<TokenStateId>* choose_successor(const LSystem* l, const std::tuple<TokenStateId, TokenStateId, int>& context);
    std::pair<TokenSize, ByteWeightedRule*> find_rule_by_probability(float p);
};

#endif //TYPES_H
