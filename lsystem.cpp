#include "lsystem.h"
#include "parse.h"

LSystem::LSystem(const std::vector<Token> &axiom,
                 TokenSet variables,
                 TokenSet constants,
                 std::map<Token, ProductionRule> parsed_rules,
                 ProbabilityDistribution *dist)
    : uniform_dist(dist),
      variables(std::move(variables)),
      constants(std::move(constants)),
      rules(std::move(parsed_rules)) {
    // Initialize byte_rules and dists
    for (int i = 0; i < 256; i++) {
        this->byte_rules[i] = nullptr;
    }
    for (int i = 0; i < 4; i++) {
        this->dists[i] = dist;
    }

    this->encode_tokens();

    this->axiom = std::vector<TokenStateId>(axiom.size());
    for (int i = 0; i < axiom.size(); i++) {
        this->axiom[i] = this->token_bytes[axiom[i]];
    }
    this->current_state = std::vector<TokenStateId>(this->axiom.size());
    this->next_state = std::vector<TokenStateId>();
    this->reset();
}

LSystem::LSystem(const std::vector<Token> &axiom,
                 const std::map<Token, std::string> &rules,
                 ProbabilityDistribution *dist) {
    auto [vars, consts, parsed_rules] = parse_rules(rules);
    new(this) LSystem(axiom,
                      std::move(vars),
                      std::move(consts),
                      std::move(parsed_rules),
                      dist);
}

void LSystem::set_dist(const int index, ProbabilityDistribution *dist) {
    this->dists[index] = dist;
}

std::vector<Token> LSystem::decode_axiom() const {
    std::vector<Token> decoded_axiom(this->axiom.size());
    for (int i = 0; i < this->axiom.size(); i++) {
        decoded_axiom[i] = this->bytes_token[this->axiom[i]];
    }
    return decoded_axiom;
}

inline float clamp(const float x, const float lower, const float upper) {
    return std::max(lower, std::min(x, upper));
}


float calculateCubicSplineProbability(const CubicSplineParams &params, float x) {
    const auto &points = params.controlPoints;

    // Find the segment containing x
    auto it = std::lower_bound(points.begin(), points.end(), x,
                               [](const auto &point, float value) { return point.first < value; });

    if (it == points.begin()) return points.front().second;
    if (it == points.end()) return points.back().second;

    auto prev = std::prev(it);
    float t = (x - prev->first) / (it->first - prev->first);

    // Linear interpolation (you might want to implement actual cubic spline interpolation for better results)
    return prev->second + t * (it->second - prev->second);
}


float LSystem::calculate_activation_probability(const ActivationStrategy &strategy, const int position) const {
    switch (strategy.type) {
        case ActivationStrategyType::Fixed:
            return strategy.fixedValue;

        case ActivationStrategyType::CubicSpline: {
            const auto &params = strategy.cubicSplineParams;
            float x = static_cast<float>(position) / current_state.size(); // Normalize position to [0, 1]
            return calculateCubicSplineProbability(params, x);
        }
        case ActivationStrategyType::Distribution: {
            const auto &params = strategy.distribution_params;
            ProbabilityDistribution *dist = this->dists[params.distribution];
            int value;
            if (std::holds_alternative<Token>(params.meta_heuristic)) {
                const auto token = std::get<Token>(params.meta_heuristic);
                value = this->param_bytes[this->token_bytes.at(token)];
            } else {
                switch (std::get<GlobalMetaHeuristic>(params.meta_heuristic)) {
                    case GlobalMetaHeuristic::Length:
                        value = this->current_state.size();
                        break;
                    case GlobalMetaHeuristic::Position:
                        value = position;
                        break;
                    case GlobalMetaHeuristic::Iteration:
                        value = this->current_iteration;
                        break;
                }
            }
            const auto [normalizing_constant, min, max, scale] = params.constants;
            if (value > normalizing_constant) value = normalizing_constant;
            return clamp(dist->cdf_bin(value, normalizing_constant) * scale, min, max);
        }
    }
    return 0.0f; // Default case, should not happen
}


void LSystem::encode_tokens() {
    this->token_bytes = std::map<Token, TokenStateId>();
    int i = 0;
    std::map<Token, TokenSize> stateful_var_params;

    for (auto &t: this->variables) {
        if (auto [base_var, number_state, is_stateful] =
                    try_parse_stateful_variable(t);
            is_stateful) {
            if (stateful_var_params.contains(base_var)) {
                stateful_var_params[base_var] = number_state;
            }
            stateful_var_params[base_var] =
                    std::max(number_state, stateful_var_params[base_var]);
        }
        const auto byte_pair = make_token_state_id(i, false);
        this->token_bytes[t] = byte_pair;
        this->bytes_token[byte_pair] = t;
        i++;
    }

    for (auto &t: this->constants) {
        const auto byte_pair = make_token_state_id(i, false);
        this->token_bytes[t] = byte_pair;
        this->bytes_token[byte_pair] = t;
        i++;
    }

    this->empty_state_id = this->token_bytes[""];

    int j = 0;
    for (auto &[base_var, max_state]: stateful_var_params) {
        constexpr int min_index = 1;
        const int max_index = max_state;
        const TokenStateId base_token_id = this->token_bytes.contains(base_var)
                                               ? this->token_bytes[base_var]
                                               : this->empty_state_id;
        for (int k = min_index; k <= max_index; k++) {
            const auto byte_pair = make_token_state_id(j, true);
            this->token_bytes[Token(base_var + std::to_string(k))] = byte_pair;
            this->bytes_token[byte_pair] = Token(base_var + std::to_string(k));
            this->param_bytes[j] = base_token_id;
            this->params[j] = k;
            j++;
        }
    }

    /// init static array of byte rules
    for (auto &[t, rule]: this->rules) {
        this->byte_rules[this->token_bytes[t]] = new ByteProductionRule();
        const auto byte_rule = this->byte_rules[this->token_bytes[t]];
        byte_rule->predecessor = this->token_bytes[t];
        byte_rule->weights = std::vector<ByteWeightedRule *>(rule.weights.size());

        rule.encode_tokens(byte_rule, this->token_bytes, this->uniform_dist, false);
    }
}

void LSystem::reset() {
    this->current_iteration = 0;
    // map axiom to bytes and copy to current state
    for (int i = 0; i < this->axiom.size(); i++) {
        this->current_state[i] = this->axiom[i];
    }
    this->current_state.resize(this->axiom.size());
}

void LSystem::iterate(const int iterations) { this->apply_rules(iterations); }

void LSystem::apply_rules(const int iterations) {
    for (int i = 0; i < iterations; i++) {
        this->apply_rules_once(this->current_state, this->next_state);
        std::swap(this->current_state, this->next_state);
        this->next_state.clear();
    }
}

void LSystem::apply_rules_once(const std::vector<TokenStateId> &input,
                               std::vector<TokenStateId> &output) {
    output.reserve(input.size() * 1.2);
    const size_t input_size = input.size();
    ++this->current_iteration;

    for (size_t i = 0; i < input_size; ++i) {
        TokenStateId token_state_id = input[i];
        TokenStateId next_token = token_state_id;
        if (has_param(token_state_id) && this->params[get_id(token_state_id)] > 1) {
            next_token--;
        }

        ByteProductionRule *rules = this->byte_rules[next_token];
        if (rules == nullptr || rules->weights.empty()) {
            output.push_back(next_token);
            continue;
        }

        TokenStateId left_context = (i > 0) ? input[i - 1] : this->empty_state_id;
        TokenStateId right_context =
                (i < input_size - 1) ? input[i + 1] : this->empty_state_id;
        std::tuple context = {left_context, right_context, i};

        const std::vector<TokenStateId> *successor =
                rules->choose_successor(this, context);
        if (successor == nullptr) {
            output.push_back(token_state_id);
        } else {
            output.insert(output.end(), successor->begin(), successor->end());
        }
    }
}
