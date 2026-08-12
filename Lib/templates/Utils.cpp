#ifndef __UTILS_CPP__
#define __UTILS_CPP__

#include "Utils.hpp"

/// Loss
    inline void
    Loss::mse(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad)
    {
        const std::size_t size = pred.size();
        for (std::size_t i = 0; i < size; ++i) {
            grad[i] = (pred[i] - expect[i]) / static_cast<double>(size);
        }
    }

    inline void
    Loss::mae(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad)
    {
        const std::size_t size = pred.size();
        for (std::size_t i = 0; i < size; ++i) {
                 if (pred[i] > expect[i]) grad[i] =  1.0 / static_cast<double>(size);
            else if (pred[i] < expect[i]) grad[i] = -1.0 / static_cast<double>(size);
            else                          grad[i] =  0.0;
        }
    }

    inline void
    Loss::huber(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad)
    {
        const std::size_t size = pred.size();
        const double delta = 1.0;
        for (std::size_t i = 0; i < size; ++i) {
            double diff = pred[i] - expect[i];
            if (std::fabs(diff) <= delta) {
                grad[i] = diff / static_cast<double>(size);
            } else {
                grad[i] = ((diff > 0.0) ? delta : -delta) / static_cast<double>(size);
            }
        }
    }

    inline void
    Loss::logCosh(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad)
    {
        const std::size_t size = pred.size();
        for (std::size_t i = 0; i < size; ++i) {
            grad[i] = std::tanh(pred[i] - expect[i]) / static_cast<double>(size);
        }
    }

    inline void
    Loss::relativeMse(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad) {
        const std::size_t size = pred.size();
        const double eps = 1e-8;
        for (std::size_t i = 0; i < size; ++i) {
            const double denom = expect[i] * expect[i] + eps;
            grad[i] = (2.0 * (pred[i] - expect[i]) / denom) / static_cast<double>(size);
        }
    }

    inline void
    Loss::bce(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad) {
        const std::size_t size = pred.size();
        const double eps = 1e-15;
        for (std::size_t i = 0; i < size; ++i) {
            double p = pred[i];
            if (p < eps) p = eps;
            if (p > 1.0 - eps) p = 1.0 - eps;
            grad[i] = ((p - expect[i]) / (p * (1.0 - p))) / static_cast<double>(size);
        }
    }

    inline void
    Loss::categoricalCrossEntropy(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad) {
        const std::size_t size = pred.size();
        const double eps = 1e-15;
        for (std::size_t i = 0; i < size; ++i) {
            double p = pred[i];
            if (p < eps) p = eps;
            grad[i] = (-expect[i] / p) / static_cast<double>(size);
        }
    }

    inline void
    Loss::hinge(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad)
    {
        const std::size_t size = pred.size();
        for (std::size_t i = 0; i < size; ++i) {
            if (expect[i] * expect[i] < 1.0) {
                grad[i] = -expect[i] / static_cast<double>(size);
            } else {
                grad[i] = 0.0;
            }
        }
    }

    inline void
    Loss::cosineProximity(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad)
    {
        const std::size_t size = pred.size();
        double dot = 0.0, normP = 0.0, normT = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            dot += pred[i] * expect[i];
            normP += pred[i] * pred[i];
            normT += expect[i] * expect[i];
        }
        normP = std::sqrt(normP) + 1e-15;
        normT = std::sqrt(normT) + 1e-15;
        const double normProduct = normP * normT;
        const double cosineSim = dot / normProduct;
        for (std::size_t i = 0; i < size; ++i) {
            grad[i] = -(expect[i] / normProduct - (cosineSim * pred[i]) / (normP * normP));
        }
    }

/// Activation
    /// raw
    inline double
    Activation::raw(double x, double alpha)
    {
        return x * alpha;
    }

    inline double
    Activation::rawDerivative(double, double alpha)
    {
        return alpha;
    }
    
    /// relu
    inline double
    Activation::relu(double x, double)
    {
        return (x > 0.0) ? x : 0.0;
    }

    inline double
    Activation::reluDerivative(double x, double)
    {
        return (x > 0.0) ? x : 0.0;
    }

    /// leakyRelu
    inline double
    Activation::leakyRelu(double x, double alpha) {
        return (x > 0.0) ? x : alpha * x;
    }

    inline double
    Activation::leakyReluDerivative(double x, double alpha) {
        return (x > 0.0) ? 1.0 : alpha;
    }

    /// relu6
    inline double
    Activation::relu6(double x, double alpha) {
        const double cap = (alpha <= 0.0) ? 6.0 : alpha;
        if (x <= 0.0) return 0.0;
        if (x >= cap) return cap;
        return x;
    }

    inline double
    Activation::relu6Derivative(double x, double alpha) {
        const double cap = (alpha <= 0.0) ? 6.0 : alpha;
        return (x > 0.0 && x < cap) ? 1.0 : 0.0;
    }

    /// elu
    inline double
    Activation::elu(double x, double alpha) {
        return (x >= 0.0) ? x : alpha * (std::exp(x) - 1.0);
    }


    inline double
    Activation::eluDerivative(double x, double) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    /// sigmoid
    inline double
    Activation::sigmoid(double x, double) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double
    Activation::sigmoidDerivative(double x, double) {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 - s);
    }

/// Init
    inline void
    Init::randomUniform(std::vector<double>& weights, std::size_t, std::size_t)
    {
        const std::size_t size = weights.size();
        for (std::size_t i = 0; i < size; ++i) {
            weights[i] = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;
        }  
    }

    inline void
    Init::xavierUniform(std::vector<double>& weights, std::size_t fanIn, std::size_t fanOut)
    {
        const std::size_t size = weights.size();
        const double limit = std::sqrt(6.0 / static_cast<double>(fanIn + fanOut));
        for (std::size_t i = 0; i < size; ++i) {
            weights[i] = (((double)std::rand() / RAND_MAX) * 2.0 - 1.0) * limit;
        }
    }

    inline void
    Init::heUniform(std::vector<double>& weights, std::size_t fanIn, std::size_t)
    {
        const std::size_t size = weights.size();
        const double limit = std::sqrt(6.0 / static_cast<double>(fanIn));
        for (std::size_t i = 0; i < size; ++i) {
            weights[i] = (((double)std::rand() / RAND_MAX) * 2.0 - 1.0) * limit;
        }
    }

    inline void
    Init::heNormal(std::vector<double>& weights, std::size_t fanIn, std::size_t)
    {
        const std::size_t size = weights.size();
        const double stddev = std::sqrt(2.0 / static_cast<double>(fanIn));    
        for (std::size_t i = 0; i < size; ++i) {
            double u1 = (static_cast<double>(std::rand()) + 1.0) / (static_cast<double>(RAND_MAX) + 1.0);
            double u2 = (static_cast<double>(std::rand()) + 1.0) / (static_cast<double>(RAND_MAX) + 1.0);
            double randNormal = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
            weights[i] = randNormal * stddev;
        }
    }

    inline void
    Init::zero(std::vector<double>& weights, std::size_t, std::size_t)
    {
        const std::size_t size = weights.size();
        for (std::size_t i = 0; i < size; ++i) {
            weights[i] = 0.0;
        }
    }

    inline void
    Init::ones(std::vector<double>& weights, std::size_t, std::size_t)
    {
        const std::size_t size = weights.size();
        for (std::size_t i = 0; i < size; ++i) {
            weights[i] = 1.0;
        }
    }

#endif /// __UTILS_cPP__
