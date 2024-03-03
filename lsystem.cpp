//
// Created by vikimaster2 on 2/18/24.
//

#include "lsystem.h"
#include "parse.h"

LSystem::LSystem(std::vector<Token> axiom, const std::map<Token, std::string>& rules, ProbabilityDistribution* dist)
    : uniform_dist(dist)
{
    auto [vars, consts, parsed_rules] = parse_rules(rules);
    this->variables = std::move(vars);
    this->constants = std::move(consts);
    this->rules = std::move(parsed_rules);

    // init to nullptr
    for (int i = 0; i < 256; i++)
    {
        this->byte_rules[i] = nullptr;
    }
    for (int i = 0; i < 4; i++)
    {
        this->dists[i] = dist;
    }
    this->encode_tokens();


    this->axiom = std::vector<TokenStateId>(axiom.size());
    for (int i = 0; i < axiom.size(); i++)
    {
        this->axiom[i] = this->token_bytes[axiom[i]];
    }
    this->current_state = std::vector<TokenStateId>(this->axiom.size());
    this->next_state = std::vector<TokenStateId>();
    this->reset();
}

void LSystem::set_dist(const int index, ProbabilityDistribution* dist)
{
    this->dists[index] = dist;
}


void LSystem::encode_tokens()
{
    this->token_bytes = std::map<Token, TokenStateId>();
    int i = 0;
    std::map<Token, TokenSize> stateful_var_params;

    for (auto& t : this->variables)
    {
        if (auto [base_var, number_state, is_stateful] = try_parse_stateful_variable(t); is_stateful)
        {
            if (stateful_var_params.contains(base_var))
            {
                stateful_var_params[base_var] = number_state;
            }
            stateful_var_params[base_var] = std::max(number_state, stateful_var_params[base_var]);
        }
        const auto byte_pair = make_token_state_id(i, false);
        this->token_bytes[t] = byte_pair;
        this->bytes_token[byte_pair] = t;
        i++;
    }

    for (auto& t : this->constants)
    {
        const auto byte_pair = make_token_state_id(i, false);
        this->token_bytes[t] = byte_pair;
        this->bytes_token[byte_pair] = t;
        i++;
    }

    this->empty_state_id = this->token_bytes[""];

    int j = 0;
    for (auto& [base_var, max_state] : stateful_var_params)
    {
        constexpr int min_index = 1;
        const int max_index = max_state;
        const TokenStateId base_token_id = this->token_bytes.contains(base_var)
                                               ? this->token_bytes[base_var]
                                               : this->empty_state_id;
        for (int k = min_index; k <= max_index; k++)
        {
            const auto byte_pair = make_token_state_id(j, true);
            this->token_bytes[Token(base_var + std::to_string(k))] = byte_pair;
            this->bytes_token[byte_pair] = Token(base_var + std::to_string(k));
            this->param_bytes[j] = base_token_id;
            this->params[j] = k;
            j++;
        }
    }

    /// init static array of byte rules
    for (auto& [t, rule] : this->rules)
    {
        this->byte_rules[this->token_bytes[t]] = new ByteProductionRule();
        const auto byte_rule = this->byte_rules[this->token_bytes[t]];
        byte_rule->predecessor = this->token_bytes[t];
        byte_rule->weights = std::vector<ByteWeightedRule*>(rule.weights.size());

        rule.encode_tokens(byte_rule, this->token_bytes, this->uniform_dist, true);
    }
}

void LSystem::reset()
{
    // map axiom to bytes and copy to current state
    for (int i = 0; i < this->axiom.size(); i++)
    {
        this->current_state[i] = this->axiom[i];
    }
    this->current_state.resize(this->axiom.size());
}


void LSystem::iterate(const int iterations)
{
    this->apply_rules(iterations);
}

void LSystem::apply_rules(const int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        this->apply_rules_once(this->current_state, this->next_state);
        std::swap(this->current_state, this->next_state);
        this->next_state.clear();
    }
}


void LSystem::apply_rules_once(const std::vector<TokenStateId>& input, std::vector<TokenStateId>& output)
{
    int i = -1;
    for (auto token_state_id : input)
    {
        i++;
        TokenStateId next_token = token_state_id;
        if (has_param(token_state_id) && this->params[get_id(token_state_id)] > 1)
        {
            next_token--;
        }

        ByteProductionRule* rules = this->byte_rules[next_token];
        if (rules == nullptr || rules->weights.empty())
        {
            output.push_back(next_token);
            continue;
        }
        std::tuple context = {this->empty_state_id, this->empty_state_id, i};
        if (i > 0)
        {
            std::get<0>(context) = input[i - 1];
        }
        if (i < input.size() - 1)
        {
            std::get<1>(context) = input[i + 1];
        }
        const std::vector<TokenStateId>* successor = rules->choose_successor(this, context);
        if (successor == nullptr)
        {
            output.push_back(token_state_id);
            continue;
        }
        output.insert(output.end(), successor->begin(), successor->end());
    }
}
