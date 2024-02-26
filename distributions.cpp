#include "distributions.h"

PresampledDistribution::PresampledDistribution(ProbabilityDistribution* dist, const int& size)
    : size(size), index(0), dist(dist)
{
    this->presampled.reserve(size);
    for (int i = 0; i < size; i++)
    {
        presampled.push_back(dist->sample());
    }
}

float PresampledDistribution::sample()
{
    if (index >= size)
    {
        index = 0;
    }
    return presampled[index++];
}

UniformDistribution::UniformDistribution()
    : min(0), max(1)
{
}

UniformDistribution::UniformDistribution(const float& min, const float& max)
    : min(min), max(max)
{
}

float UniformDistribution::sample()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}


KumaraswamyDistribution::KumaraswamyDistribution(const float& alpha, const float& beta)
    : alpha(alpha), beta(beta), uniform_distribution(new UniformDistribution(0, 1))
{
}

float KumaraswamyDistribution::sample()
{
    const float u = uniform_distribution->sample();
    return std::pow(1 - std::pow(u, 1 / beta), 1 / alpha);
}
