#ifndef __NETWORK_HPP__
#define __NETWORK_HPP__
#include "Layer.hpp"
#include <vector>

class
Network
{
public:
    Network(const std::vector<size_t>& topology,
            const std::vector<double(*)(double,double)>& elus,
            const std::vector<double(*)(double,double)>& eluDerivatives,
            const std::vector<double>& eluAlphas,
            const std::vector<double>& eluDerivativeAlphas,
            void(*init)(std::vector<double>&,size_t,size_t),
            void(*loss)(const std::vector<double>&, const std::vector<double>&, std::vector<double>&),
            const double               learnRate);
    
    void train(std::vector<double>& input, std::vector<double>& output);
    const std::vector<double>& run(std::vector<double>& input);

private:    
    std::vector<double>& forward(std::vector<double>& inputs);
    void        backward(const std::vector<double>& initialGradients);

private:
    std::vector<Layer> layers_;
    void(*loss_)(const std::vector<double>&, const std::vector<double>&, std::vector<double>&);
    double learnRate_;
    std::vector<double> initGradient_;
};

#endif /// __NETWORK_HPP__
