#include "types.h"

#include "lsystem.h"

inline float clamp(const float x, const float lower, const float upper) {
    return std::max(lower, std::min(x, upper));
}

std::vector<Token> ProductionRule::choose_successor() const {
    float total_weight = 0.0;
    for (auto &rule: this->weights)
        total_weight += rule.weight;

    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float random_weight = r * total_weight;
    for (auto &rule: this->weights) {
        random_weight -= rule.weight;
        if (random_weight <= 0)
            return rule.products;
    }

    return {};
}


void ProductionRule::encode_tokens(ByteProductionRule *rule, std::map<Token, TokenStateId> token_bytes,
                                   ProbabilityDistribution *dist,
                                   const bool &presample) const {
    float total = 0.0;
    for (int w = 0; w < this->weights.size(); w++) {
        WeightedRule wt = this->weights[w];
        std::vector<TokenStateId> encodedTokens(wt.products.size());
        for (int i = wt.products.size() - 1; i >= 0; i--) {
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
            wt.activation_strategy,
            encodedTokens,
        };
        total += wt.weight;
    }
    if (presample) {
        rule->pre_sample(dist);
    }
}

void ByteProductionRule::pre_sample(ProbabilityDistribution *dist) {
    if (this->weights.empty()) {
        return;
    }
    if (this->preSampledWeights.empty()) {
        this->preSampledWeights = std::vector<TokenSize>(25600);
    }
    for (int i = 0; i < 25600; i++) {
        const float random = dist->sample() * this->weights[this->weights.size() - 1]->upper_limit;
        auto [index, _] = this->find_rule_by_probability(random);
        this->preSampledWeights[i] = index;
    }
}

std::pair<TokenSize, ByteWeightedRule *> ByteProductionRule::find_rule_by_probability(float p) {
    int lo = 0;
    int hi = this->weights.size();
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (p < this->weights[mid]->lower_limit) {
            hi = mid;
        } else if (p >= this->weights[mid]->upper_limit) {
            lo = mid + 1;
        } else {
            return {mid, this->weights[mid]};
        }
    }
    return {0, nullptr};
}

const std::vector<TokenStateId> *ByteProductionRule::choose_successor(
    const LSystem *l, const std::tuple<TokenStateId, TokenStateId, int> &context) {
    const TokenStateId emptyToken = l->empty_state_id;
    TokenStateId previousToken = std::get<0>(context);
    TokenStateId nextToken = std::get<1>(context);
    const int position = std::get<2>(context);

    if (has_param(previousToken)) {
        previousToken = l->param_bytes[get_id(previousToken)];
    }
    if (has_param(nextToken)) {
        nextToken = l->param_bytes[get_id(nextToken)];
    }

    if (!this->preSampledWeights.empty()) {
        const ByteWeightedRule *rule = this->weights[this->preSampledWeights[this->currentIndex]];
        this->currentIndex = (this->currentIndex + 1) % this->preSampledWeights.size();

        if (rule->catalyst == emptyToken ||
            (rule->catalyst_position == CatalystPosition::Left && previousToken == rule->catalyst) ||
            (rule->catalyst_position == CatalystPosition::Right && nextToken == rule->catalyst)) {
            float activation_prob = l->calculate_activation_probability(rule->activation_strategy, position);
            if (l->uniform_dist->sample() < activation_prob) {
                return &rule->products;
            }
        }
        return nullptr;
    }

    const float random = l->uniform_dist->sample() * this->weights[this->weights.size() - 1]->upper_limit;
    auto [_, rule] = this->find_rule_by_probability(random);
    if (rule == nullptr) {
        return nullptr;
    }

    if (rule->catalyst == emptyToken ||
        (rule->catalyst_position == CatalystPosition::Left && previousToken == rule->catalyst) ||
        (rule->catalyst_position == CatalystPosition::Right && nextToken == rule->catalyst)) {
        float activation_prob = l->calculate_activation_probability(rule->activation_strategy, position);
        if (l->uniform_dist->sample() < activation_prob) {
            return &rule->products;
        }
    }
    return nullptr;
}
