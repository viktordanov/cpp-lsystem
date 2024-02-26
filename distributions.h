//
// Created by vikimaster2 on 2/18/24.
//

#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <random>
#include <vector>

class ProbabilityDistribution
{
public:
    virtual ~ProbabilityDistribution() = default;
    virtual float sample() = 0;
};


class PresampledDistribution final : public ProbabilityDistribution
{
    int size;
    int index;

    ProbabilityDistribution* dist;
    std::vector<float> presampled;

public:
    PresampledDistribution(ProbabilityDistribution* dist, const int& size);
    float sample() override;
};



class UniformDistribution final : public ProbabilityDistribution
{
    float min;
    float max;

public:
    UniformDistribution();
    UniformDistribution(const float& min, const float& max);
    float sample() override;
};

class KumaraswamyDistribution final : public ProbabilityDistribution
{
    float alpha;
    float beta;
    ProbabilityDistribution* uniform_distribution;


public:

    KumaraswamyDistribution(const float& alpha, const float& beta);
    float sample() override;
};


#endif //DISTRIBUTIONS_H
