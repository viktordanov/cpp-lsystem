#ifndef CUDA_LSYSTEM_CPP_DEBUG_KL_DIVERGENCE_H
#define CUDA_LSYSTEM_CPP_DEBUG_KL_DIVERGENCE_H

// Type aliases
#include <cmath>
#include <numeric>
#include "types.h"

using Distribution = std::map<Token, double>;

inline double kl_divergence(const Distribution& p, const Distribution& q)
{
    double kl_div = 0.0;
    for (const auto& [token, p_value] : p)
    {
        if (q.count(token) > 0 && p_value > 0)
        {
            double q_value = q.at(token);
            kl_div += p_value * std::log(p_value / q_value);
        }
    }
    return kl_div;
}

Distribution get_distribution(const LSystem& l_system)
{
    Distribution dist;
    for (const auto& token_state_id : l_system.current_state_bytes())
    {
        std::string token = l_system.bytes_token[token_state_id];
        dist[token]++;
    }
    double total = std::accumulate(dist.begin(), dist.end(), 0.0,
                                   [](double sum, const auto& pair) { return sum + pair.second; });
    for (auto& [token, count] : dist)
    {
        count /= total;
    }
    return dist;
}

Distribution create_target_distribution(int n)
{
    Distribution dist;
    dist["l"] = 1.0 / (n - 1);
    dist["L"] = 1.0 - dist["l"];
    return dist;
}

#endif //CUDA_LSYSTEM_CPP_DEBUG_KL_DIVERGENCE_H
