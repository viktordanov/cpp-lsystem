//
// Created by vikimaster2 on 2/18/24.
//

#ifndef PARSE_H
#define PARSE_H
#include <string>
#include <vector>

#include "strings.h"
#include "types.h"

inline bool is_capitalized(const Token& t)
{
    if (t.empty())
    {
        return false;
    }
    return t[0] >= 'A' && t[0] <= 'Z';
}

inline bool is_variable(const Token& t)
{
    if (t.empty())
    {
        return false;
    }
    return is_capitalized(t) && t[t.size() - 1] != '_';
}

inline std::vector<Token> symbols_to_tokens(const std::vector<Token>& symbols)
{
    std::vector<Token> tokens;
    for (auto& s : symbols)
    {
        tokens.push_back(s);
    }
    return tokens;
}

inline std::vector<WeightedRule> parse_rule(const std::string& str)
{
    std::vector<WeightedRule> weightedTokens;
    std::vector<std::string> groups;
    std::string group;
    for (auto& c : str)
    {
        if (c == ';')
        {
            trim(group);
            groups.push_back(group);
            group.clear();
        }
        else
        {
            group.push_back(c);
        }
    }
    if (!group.empty())
    {
        groups.push_back(group);
    }

    // remove tailing and leading spaces from groups
    for (auto& group : groups)
    {
        while (!group.empty() && group[0] == ' ')
        {
            group = group.substr(1);
        }
        while (!group.empty() && group[group.size() - 1] == ' ')
        {
            group = group.substr(0, group.size() - 1);
        }
    }

    for (auto& token_group : groups)
    {
        if (token_group.empty())
        {
            continue;
        }
        std::vector<std::string> tokens;
        std::string token;
        for (const auto& c : token_group)
        {
            if (c == ' ')
            {
                tokens.push_back(token);
                token.clear();
            }
            else
            {
                token.push_back(c);
            }
        }
        if (!token.empty())
        {
            tokens.push_back(token);
        }

        const float weight = std::stof(tokens[0]);
        if (tokens.size() > 1 && tokens[1][0] == '*')
        {
            weightedTokens.push_back(WeightedRule{
                weight,
                tokens[1].substr(1),
                symbols_to_tokens(std::vector(tokens.begin() + 2, tokens.end()))
            });
            continue;
        }
        weightedTokens.push_back(WeightedRule{
            weight,
            "",
            symbols_to_tokens(std::vector(tokens.begin() + 1, tokens.end()))
        });
    }
    return weightedTokens;
}

inline std::tuple<TokenSet, TokenSet, std::map<Token, ProductionRule>> parse_rules(const std::map<Token, std::string>& rulesMap)
{
    TokenSet vars;
    TokenSet consts;
    std::map<Token, ProductionRule> parsedRules;


    const auto index_token = [&vars, &consts](const Token& token)
    {
        if (is_variable(token))
        {
            vars.insert(token);
        }
        else
        {
            consts.insert(token);
        }
    };

    for (auto& [key, value] : rulesMap)
    {
        index_token(key);
        parsedRules[key] = ProductionRule{key, parse_rule(value)};

        for (auto& [weight, catalyst, products] : parsedRules[key].weights)
        {
            index_token(catalyst);
            for (auto& token : products)
            {
                index_token(token);
            }
        }
    }

    return {vars, consts, parsedRules};
}


inline std::tuple<Token, TokenSize, bool> try_parse_stateful_variable(const Token& t)
{
    std::string variable;
    TokenSize num = 0;
    std::string sb;
    TokenSize cumulative_number = 0;

    if (t[t.size() - 1] < '0' || t[t.size() - 1] > '9')
    {
        return std::make_tuple(Token(""), 0, false);
    }

    for (const auto r : t)
    {
        if (r >= '0' && r <= '9')
        {
            cumulative_number = cumulative_number * 10 + (r - '0');
            continue;
        }
        if (cumulative_number == 0)
        {
            sb.push_back(r);
        }
    }

    if (cumulative_number == 0)
    {
        return std::make_tuple(Token(""), 0, false);
    }
    if (cumulative_number > 255)
    {
        cumulative_number = 255;
    }
    num = cumulative_number;

    return std::make_tuple(Token(sb), num, true);
}

#endif //PARSE_H
