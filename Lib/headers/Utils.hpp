#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <vector>
#include <cmath>
#include <cstdlib>

namespace
Loss
{
    inline void mse(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad);
    inline void mae(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad);
    inline void huber(const std::vector<double>& pred, const std::vector<double>& xpect, std::vector<double>& grad);
    inline void logCosh(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad);
    inline void relativeMse(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad);
    inline void bce(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad);
    inline void categoricalCrossEntropy(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad);
    inline void hinge(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad);
    inline void cosineProximity(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad);
}

namespace
Activation
{
    /// relu
    inline double raw(double x,double);
    inline double rawDerivative(double x,double);

    /// relu
    inline double relu(double x,double);
    inline double reluDerivative(double x,double);

    /// leakyRelu
    inline double leakyRelu(double x,double alpha);
    inline double leakyReluDerivative(double x,double alpha);

    /// relu6
    inline double relu6(double x, double alpha);
    inline double relu6Derivative(double x, double alpha);

    /// elu
    inline double elu(double x, double alpha);
    inline double eluDerivative(double x, double alpha);

    /// sigmoid
    inline double sigmoid(double x, double);
    inline double sigmoidDerivative(double x, double);
}

namespace
Init
{
    inline void randomUniform(std::vector<double>& weights, std::size_t, std::size_t);
    inline void xavierUniform(std::vector<double>& weights, std::size_t fanIn, std::size_t fanOut);
    inline void heUniform(std::vector<double>& weights, std::size_t fanIn, std::size_t);
    inline void heNormal(std::vector<double>& weights, std::size_t fanIn, std::size_t);
    inline void zero(std::vector<double>& weights, std::size_t, std::size_t);
    inline void ones(std::vector<double>& weights, std::size_t, std::size_t);
}


#include "../templates/Utils.cpp"
#endif /// __UTILS_HPP__
