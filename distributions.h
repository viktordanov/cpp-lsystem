//
// Created by vikimaster2 on 2/18/24.
//

#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <map>
#include <vector>

class ProbabilityDistribution
{
public:
    virtual ~ProbabilityDistribution() = default;
    virtual float sample() = 0;
    virtual float pdf(const float& x) = 0;
    virtual float cdf(const float& x) = 0;
    virtual float cdf_bin(const int& x, const float& bins) = 0;
};


class PresampledDistribution final : public ProbabilityDistribution
{
    int size;
    int index;

    ProbabilityDistribution* dist;
    std::vector<float> presampled;
    float** cdf_bin_cache;

public:
    PresampledDistribution(ProbabilityDistribution* dist, const int& size);
    float sample() override;
    float pdf(const float& x) override;
    float cdf(const float& x) override;
    float cdf_bin(const int& x, const float& bins) override;

    void presample_cdf_bin(const float& bins);
};


class ScaledProbabilityDistribution final : public ProbabilityDistribution
{
    float scale;
    ProbabilityDistribution* dist;

public:
    ScaledProbabilityDistribution(const float& scale, ProbabilityDistribution* dist);
    float sample() override;
    float pdf(const float& x) override;
    float cdf(const float& x);
    float cdf_bin(const int& x, const float& bins);
};


class UniformDistribution final : public ProbabilityDistribution
{
    float min;
    float max;

public:
    UniformDistribution();
    UniformDistribution(const float& min, const float& max);
    float sample() override;
    float pdf(const float& x) override;
    float cdf(const float& x);
    float cdf_bin(const int& x, const float& bins);
};

class KumaraswamyDistribution final : public ProbabilityDistribution
{
    float alpha;
    float beta;
    ProbabilityDistribution* uniform_distribution;

public:
    KumaraswamyDistribution(const float& alpha, const float& beta);
    float sample() override;
    float pdf(const float& x) override;
    float cdf(const float& x);
    float cdf_bin(const int& x, const float& bins);
};


#endif //DISTRIBUTIONS_H
