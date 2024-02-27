#include "types.h"

#include <bits/stl_algo.h>

#include "lsystem.h"

inline float clamp(const float x, const float lower, const float upper)
{
    return std::max(lower, std::min(x, upper));
}

std::vector<Token> ProductionRule::choose_successor() const
{
    float total_weight = 0.0;
    for (auto& rule : this->weights)
        total_weight += rule.weight;

    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float random_weight = r * total_weight;
    for (auto& rule : this->weights)
    {
        random_weight -= rule.weight;
        if (random_weight <= 0)
            return rule.products;
    }

    return {};
}


void ProductionRule::encode_tokens(ByteProductionRule* rule, std::map<Token, TokenStateId> token_bytes,
                                   ProbabilityDistribution* dist,
                                   const bool& presample) const
{
    float total = 0.0;
    for (int w = 0; w < this->weights.size(); w++)
    {
        WeightedRule wt = this->weights[w];
        std::vector<TokenStateId> encodedTokens(wt.products.size());
        for (int i = wt.products.size() - 1; i >= 0; i--)
        {
            Token t = wt.products[i];
            encodedTokens[i] = token_bytes[t];
        }
        const TokenStateId catalyst = token_bytes.contains(wt.catalyst) ? token_bytes[wt.catalyst] : TokenStateId();
        // TODO: may cause segfault
        rule->weights[w] = new ByteWeightedRule{
            total,
            total + wt.weight,
            wt.catalyst_position,
            catalyst,
            wt.activation_probability,
            encodedTokens,
        };
        total += wt.weight;
    }
    if (presample)
    {
        rule->pre_sample(dist);
    }
}

void ByteProductionRule::pre_sample(ProbabilityDistribution* dist)
{
    if (this->weights.empty())
    {
        return;
    }
    if (this->preSampledWeights.empty())
    {
        this->preSampledWeights = std::vector<TokenSize>(25600);
    }
    for (int i = 0; i < 25600; i++)
    {
        const float random = dist->sample() * this->weights[this->weights.size() - 1]->upper_limit;
        auto [index, _] = this->find_rule_by_probability(random);
        this->preSampledWeights[i] = index;
    }
}

std::pair<TokenSize, ByteWeightedRule*> ByteProductionRule::find_rule_by_probability(float p)
{
    int lo = 0;
    int hi = this->weights.size();
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (p < this->weights[mid]->lower_limit)
        {
            hi = mid;
        }
        else if (p >= this->weights[mid]->upper_limit)
        {
            lo = mid + 1;
        }
        else
        {
            return {mid, this->weights[mid]};
        }
    }
    return {0, nullptr};
}

const std::vector<TokenStateId>* ByteProductionRule::choose_successor(
    LSystem* l, const std::pair<TokenStateId, TokenStateId>& context)
{
    const TokenStateId emptyToken = l->empty_state_id;
    TokenStateId previousToken = context.first;
    TokenStateId nextToken = context.second;

    const auto calculate_named_rule_activation = [](const NamedActivationProbability& named_activation,
                                                    LSystem* l) -> float
    {
        ProbabilityDistribution* dist = l->dists[named_activation.distribution];
        int value;
        if (std::holds_alternative<Token>(named_activation.meta_heuristic))
        {
            const auto token = std::get<Token>(named_activation.meta_heuristic);
            value = l->param_bytes[l->token_bytes[token]];
        }
        else
        {
            switch (std::get<GlobalMetaHeuristic>(named_activation.meta_heuristic))
            {
            case GlobalMetaHeuristic::Length:
                value = l->current_state.size();
                break;
            default:
                value = 0;
            }
        }

        const auto [normalizing_constant, min, max, scale] = named_activation.probability_shape_constants;
        return std::clamp(dist->cdf_bin(value, normalizing_constant)*scale, min, max);
    };

    const auto calculate_fixed_rule_activation = [](const FixedActivationProbability& fixed_activation) -> float
    {
        return clamp(fixed_activation.value, 0, 1);
    };

    const auto check_activation = [&](const ActivationProbability& activation_probability, LSystem* l) -> bool
    {
        float probability_of_activation;
        if (std::holds_alternative<FixedActivationProbability>(activation_probability))
        {
            probability_of_activation = calculate_fixed_rule_activation(
                std::get<FixedActivationProbability>(activation_probability));
        }
        else
        {
            probability_of_activation = calculate_named_rule_activation(
                std::get<NamedActivationProbability>(activation_probability), l);
        }

        const float rand = l->uniform_dist->sample();
        return rand < probability_of_activation;
    };

    if (has_param(previousToken))
    {
        previousToken = l->param_bytes[get_id(previousToken)];
    }
    if (has_param(nextToken))
    {
        nextToken = l->param_bytes[get_id(nextToken)];
    }

    if (!this->preSampledWeights.empty())
    {
        const ByteWeightedRule* rule = this->weights[this->preSampledWeights[this->currentIndex]];
        this->currentIndex++;
        if (this->currentIndex == this->preSampledWeights.size())
        {
            this->currentIndex = 0;
        }
        if (rule->catalyst == emptyToken || (rule->catalyst_position == CatalystPosition::Left && previousToken == rule
                ->catalyst) ||
            (rule->catalyst_position == CatalystPosition::Right && nextToken == rule->catalyst))
        {
            if (!check_activation(rule->activation_probability, l)) return nullptr;
            return &rule->products;
        }
        return nullptr;
    }
    const float random = l->uniform_dist->sample() * this->weights[this->weights.size() - 1]->upper_limit;
    auto [_, rule] = this->find_rule_by_probability(random);
    if (rule == nullptr)
    {
        return nullptr;
    }

    if (rule->catalyst == emptyToken || (rule->catalyst_position == CatalystPosition::Left && previousToken == rule->
            catalyst) ||
        (rule->catalyst_position == CatalystPosition::Right && nextToken == rule->catalyst))
    {
        if (!check_activation(rule->activation_probability, l)) return nullptr;
        return &rule->products;
    }
    {
        return &rule->products;
    }
    return nullptr;
}
