#include "distributions.h"

#include <cmath>
#include <random>

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

float PresampledDistribution::pdf(const float& x)
{
    return dist->pdf(x);
}

float PresampledDistribution::cdf(const float& x)
{
    return dist->cdf(x);
}

float PresampledDistribution::cdf_bin(const int& x, const float& bins)
{
    if (index >= size)
    {
        index = 0;
    }
    if (cdf_bin_cache == nullptr)
    {
        presample_cdf_bin(bins);
    }
    return cdf_bin_cache[x][index++];
}

void PresampledDistribution::presample_cdf_bin(const float& bins)
{
    this->cdf_bin_cache = new float*[bins];

    for (int i = 0; i <= bins; i++)
    {
        this->cdf_bin_cache[i] = new float[size];
        for (int j = 0; j < size; j++)
        {
            this->cdf_bin_cache[i][j] = dist->cdf_bin(i, bins);
        }
    }
}

ScaledProbabilityDistribution::ScaledProbabilityDistribution(const float& scale, ProbabilityDistribution* dist)
    : scale(scale), dist(dist)
{
}

float ScaledProbabilityDistribution::sample()
{
    return dist->sample() * scale;
}

float ScaledProbabilityDistribution::pdf(const float& x)
{
    return dist->pdf(x) * scale;
}

float ScaledProbabilityDistribution::cdf(const float& x)
{
    return dist->cdf(x) * scale;
}

float ScaledProbabilityDistribution::cdf_bin(const int& x, const float& bins)
{
    return dist->cdf_bin(x, bins) * scale;
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

float UniformDistribution::pdf(const float& x)
{
    return 1 / (max - min);
}

float UniformDistribution::cdf(const float& x)
{
    return (x - min) / (max - min);
}

float UniformDistribution::cdf_bin(const int& x, const float& bins)
{
    const float bin_width = (max - min) / bins;
    const float bin_start = x * bin_width;
    const float bin_end = (x + 1) * bin_width;
    return cdf(bin_end) - cdf(bin_start);
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

float KumaraswamyDistribution::pdf(const float& x)
{
    return alpha * beta * std::pow(x, alpha - 1) * std::pow(1 - std::pow(x, alpha), beta - 1);
}

float KumaraswamyDistribution::cdf(const float& x)
{
    return 1 - std::pow(1 - std::pow(x, alpha), beta);
}

float KumaraswamyDistribution::cdf_bin(const int& x, const float& bins)
{
    const float bin_width = 1.0 / bins;
    const float bin_start = x * bin_width;
    const float bin_end = (x + 1) * bin_width;
    return cdf(bin_end) - cdf(bin_start);
}