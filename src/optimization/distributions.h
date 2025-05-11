//
// Created by vikimaster2 on 2/18/24.
//

#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H


#include <iostream>
#include <iomanip>
#include <vector>

class ProbabilityDistribution
{
public:
    virtual ~ProbabilityDistribution() = default;
    virtual float sample() = 0;
    virtual float pdf(const float& x) = 0;
    virtual float cdf(const float& x) = 0;
    virtual float cdf_bin(const int& x, const float& bins) = 0;

 std::ostream& visualize(std::ostream& os, int width = 60, int height = 20) {
        std::vector<float> values(width);
        float max_value = 0;

        // Calculate PDF values and find the maximum
        for (int i = 0; i < width; ++i) {
            float x = static_cast<float>(i) / (width - 1);
            values[i] = pdf(x);
            max_value = std::max(max_value, values[i]);
        }

        // Normalize values and create the plot
        for (int h = height - 1; h >= 0; --h) {
            os << '|';
            for (int w = 0; w < width; ++w) {
                float normalized = values[w] / max_value;
                os << (normalized > static_cast<float>(h) / (height - 1) ? '#' : ' ');
            }
            os << '|' << std::endl;
        }

        // X-axis
        os << '+' << std::string(width, '-') << '+' << std::endl;

        // X-axis labels
        os << "0" << std::string(width - 5, ' ') << "0.5" << std::string(width - 5, ' ') << "1" << std::endl;

        // Y-axis label (max value)
        os << "Max: " << std::fixed << std::setprecision(4) << max_value << std::endl;

        return os;
    }};


class PresampledDistribution final : public ProbabilityDistribution
{
    int size;
    int index;
    int bin_count{0};

    ProbabilityDistribution* dist;
    std::vector<float> presampled;
    float** cdf_bin_cache{nullptr};

public:
    PresampledDistribution(ProbabilityDistribution* dist, const int& size);
    ~PresampledDistribution();

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
    float cdf(const float& x) override;
    float cdf_bin(const int& x, const float& bins) override;
};


class UniformDistribution final : public ProbabilityDistribution
{
    float min;
    float max;

public:
    UniformDistribution();
    ~UniformDistribution() override = default;
    UniformDistribution(const float& min, const float& max);
    float sample() override;
    float pdf(const float& x) override;
    float cdf(const float& x) override;
    float cdf_bin(const int& x, const float& bins) override;
};

class KumaraswamyDistribution final : public ProbabilityDistribution
{
    float alpha;
    float beta;
    ProbabilityDistribution* uniform_distribution;

public:
    KumaraswamyDistribution(const float& alpha, const float& beta);
    ~KumaraswamyDistribution();
    float sample() override;
    float pdf(const float& x) override;
    float cdf(const float& x) override;
    float cdf_bin(const int& x, const float& bins) override;
};


#endif //DISTRIBUTIONS_H
