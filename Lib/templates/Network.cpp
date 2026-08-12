#ifndef __NETWORK_CPP__
#define __NETWORK_CPP__
#include "Network.hpp"
#include <iostream>

Network::Network(const std::vector<size_t>& topology,
                 const std::vector<double(*)(double,double)>& elus,
                 const std::vector<double(*)(double,double)>& eluDerivatives,
                 const std::vector<double>& eluAlphas,
                 const std::vector<double>& eluDerivativeAlphas,
                 void(*init)(std::vector<double>&,size_t,size_t),
                 void(*loss)(const std::vector<double>&, const std::vector<double>&, std::vector<double>&),
                 const double learnRate)
    : loss_(loss), learnRate_(learnRate)
{
    if (topology.size() < 2) return;

    layers_.reserve(topology.size() - 1);
    for (std::size_t i = 1; i < topology.size(); ++i) {
        const std::size_t inputsForThisLayer = topology[i - 1];
        const std::size_t nodesInThisLayer = topology[i];
        
        layers_.push_back(Layer(inputsForThisLayer, 
                                NULL,
                                nodesInThisLayer, 
                                elus[i - 1], 
                                eluAlphas[i - 1], 
                                eluDerivatives[i - 1], 
                                eluDerivativeAlphas[i - 1], 
                                init));
    }
    for (std::size_t i = 1; i < layers_.size(); ++i) {
        layers_[i].setInputSource(layers_[i - 1].returnOut());
    }
    for (std::size_t i = 0; i < layers_.size() - 1; ++i) {
        layers_[i].setGradientSource(layers_[i + 1].returnGrad());
    }
    initGradient_.resize(topology.back());
    std::cout << "Network Created successfully." << std::endl;
}

std::vector<double>&
Network::forward(std::vector<double>& input)
{
    layers_[0].setInputSource(&input);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].compute();
    }
    return *layers_.back().returnOut();
}

void
Network::backward(const std::vector<double>& initialGradients)
{
    layers_.back().setGradientSource(&initialGradients);
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        layers_[static_cast<std::size_t>(i)].backpropagate(learnRate_);
    }
}

const std::vector<double>&
Network::run(std::vector<double>& input)
{
    layers_[0].setInputSource(&input);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].predict();
    }
    return *layers_.back().returnOut(); // Returns reference: Zero allocation!
}

void
Network::train(std::vector<double>& input, std::vector<double>& output)
{
    std::vector<double>& prediction = forward(input);
    loss_(prediction, output, initGradient_);
    backward(initGradient_);
}

#endif /// __NETWORK_CPP__
