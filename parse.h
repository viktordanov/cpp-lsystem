//
// Created by vikimaster2 on 2/18/24.
//

#ifndef PARSE_H
#define PARSE_H
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <charconv>

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

inline std::array<float, 4> parseCommaDelimitedFloats(const std::string& probStr) {
    std::array<float, 4> values;
    size_t start = 0, end = 0;
    int idx = 0;

    while ((end = probStr.find(',', start)) != std::string::npos && idx < 3) { // Ensure idx < 3 for last value handling
        values[idx++] = std::stof(probStr.substr(start, end - start));
        start = end + 1;
    }
    // Parse the last (or only) value in the string
    values[idx] = std::stof(probStr.substr(start, end));

    return values;
}


inline std::vector<std::pair<float, float>> parseControlPoints(const std::string& str) {
    std::vector<std::pair<float, float> > points;
    std::istringstream ss(str);
    char ch;
    float x, y;
    while (ss >> ch >> x >> ch >> y >> ch) {
        points.emplace_back(x, y);
    }
    return points;
}

inline ActivationStrategy parseActivationStrategy(const std::string& probStr) {
    if (isdigit(probStr[0])) {
        return ActivationStrategy(std::stof(probStr));
    } else if (probStr[0] == '[') {
        return ActivationStrategy(parseControlPoints(probStr));
    } else {
        const auto openBracketPos = probStr.find('[');
        assert(openBracketPos != std::string::npos);

        const auto commaPos = probStr.find(',');
        assert(commaPos != std::string::npos);

        int distribution = std::stoi(probStr.substr(2, openBracketPos - 2));

        std::variant<Token, GlobalMetaHeuristic> metaHeuristic;
        std::string metaHeuristicStr = probStr.substr(openBracketPos + 1, commaPos - openBracketPos - 1);

        if (metaHeuristicStr.starts_with('&')) {
            // It's a GlobalMetaHeuristic
            int metaHeuristicValue = std::stoi(metaHeuristicStr.substr(1));
            metaHeuristic = static_cast<GlobalMetaHeuristic>(metaHeuristicValue);
        } else {
            // It's a Token
            metaHeuristic = Token(metaHeuristicStr);
        }

        std::array<float, 4> constants = parseCommaDelimitedFloats(probStr.substr(commaPos + 1, probStr.size() - commaPos - 2));

        return ActivationStrategy(DistributionParams{distribution, metaHeuristic, constants});
    }
}


// New parse function
inline std::vector<WeightedRule> parse_rule(const std::string& str)
{
    std::vector<WeightedRule> weightedRules;
    std::istringstream stream(str);
    std::string line;

    while (std::getline(stream, line, ';'))
    {
        std::istringstream linestream(line);
        float weight;
        linestream >> weight;

        std::string token;
        bool isCatalystFound = false;
        bool isCatalystInitialized = false;
        auto catalystPosition = CatalystPosition::None;
        Token catalyst;
        ActivationStrategy activationStrategy = ActivationStrategy{1.0f};
        std::vector<Token> products;

        while (linestream >> token)
        {
            // Check for catalyst and activation probability
            size_t colonPos = token.find(':');

            if (size_t asteriskPos = token.find('*'); asteriskPos != std::string::npos)
            {
                isCatalystFound = true;
                catalystPosition = (asteriskPos == 0) ? CatalystPosition::Left : CatalystPosition::Right;
                catalyst = (asteriskPos == 0) ? token.substr(1, colonPos - 1) : token.substr(0, asteriskPos);
                if (catalyst.empty())
                {
                    throw std::invalid_argument("Invalid rule format");
                }
            }

            // Extract activation probability if present
            if (colonPos != std::string::npos)
            {
                isCatalystInitialized = true;
                std::string probStr = token.substr(colonPos + 1);
                activationStrategy = parseActivationStrategy(probStr);
            }
            // Extract when only catalyst is present
            else if (isCatalystFound)
            {
                if (isCatalystInitialized)
                {
                    products.push_back(token);
                }
                else
                {
                    activationStrategy = ActivationStrategy{1.0f};
                    isCatalystInitialized = true;
                }
            }
            else {
                // Handle adding to products
                products.push_back(token);
            }
        }

        // Construct the WeightedRule object
        weightedRules.push_back(WeightedRule{
            weight,
            catalystPosition,
            catalyst,
            activationStrategy,
            products
        });
    }

    return weightedRules;
}

inline std::tuple<TokenSet, TokenSet, std::map<Token, ProductionRule>> parse_rules(
    const std::map<Token, std::string>& rulesMap)
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

        // float weight;
        // CatalystPosition catalyst_position = CatalystPosition::None;
        // Token catalyst;
        // ActivationProbability activation_probability;
        // std::vector<Token> products;
        // size_t num_products = 0;
        for (auto& [weight, catalystPosition, catalyst, activationProbability, products] : parsedRules[key].weights)
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
