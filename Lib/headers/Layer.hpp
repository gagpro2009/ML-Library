#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include <cstdlib>
#include <cassert>
#include <vector>

class Layer
{
public:
    Layer(const std::size_t inputCount, 
          std::vector<double>* prevOut, 
          const std::size_t nodeCount, 
          double(*elu)(double, double), 
          const double eluAlpha, 
          double(*eluDerivative)(double, double), 
          const double eluDerivativeAlpha, 
          void(*init)(std::vector<double>&, size_t, size_t));

    void compute();
    void predict();
    void backpropagate(const double learningRate);
    std::vector<double>*       returnOut();
    std::vector<double>*       returnGrad();
    const std::vector<double>* returnOut() const;
    const std::vector<double>* returnGrad() const;

    void setInputSource(std::vector<double>* source);
    void setGradientSource(const std::vector<double>* source);

private:
    std::vector<double> properties_;

    double(*elu_)(double, double);
    double eluAlpha_;

    double(*eluDerivative_)(double, double);
    double eluDerivativeAlpha_;
    
    std::vector<double>*       lastInputs_;
    const std::vector<double>* downstreamGradientsSource_;

    std::vector<double> lastZs_;
    std::vector<double> lastOutputs_;
    std::vector<double> backGradients_;
    std::vector<double> deltas_;
};

#endif /// __LAYER_HPP__