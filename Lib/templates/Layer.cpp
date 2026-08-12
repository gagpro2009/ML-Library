#ifndef __LAYER_CPP__
#define __LAYER_CPP__
#include "Layer.hpp"

Layer::Layer(const std::size_t inputCount,
             std::vector<double>* prevOut,
             const std::size_t nodeCount,
             double(*elu)(double,double),
             const double eluAlpha,
             double(*eluDerivative)(double,double),
             const double eluDerivativeAlpha,
             void(*init)(std::vector<double>&,size_t,size_t))
    :elu_(elu), eluAlpha_(eluAlpha), eluDerivative_(eluDerivative), eluDerivativeAlpha_(eluDerivativeAlpha), lastInputs_(prevOut), downstreamGradientsSource_(NULL)
{
    properties_.resize( (inputCount + 1) * nodeCount);
    lastZs_.resize(nodeCount);
    lastOutputs_.resize(nodeCount);
    deltas_.resize(nodeCount);
    backGradients_.resize(inputCount);
    init(properties_, inputCount, nodeCount);
}

void
Layer::compute()
{
    const size_t nodeCount = lastOutputs_.size();
    const size_t rowSize = (*lastInputs_).size() + 1;

    for (size_t i = 0; i < nodeCount; ++i) {
        double nodeResult = 0;
        for (size_t j = 1; j < rowSize; ++j) {
            nodeResult += properties_[i * rowSize + j] * (*lastInputs_)[j - 1];
        }
        nodeResult += properties_[i * rowSize];;
        lastZs_[i] = nodeResult;
        lastOutputs_[i] = elu_(nodeResult, eluAlpha_);
    }
}

void
Layer::predict()
{
    const size_t nodeCount = lastOutputs_.size();
    const size_t rowSize = (*lastInputs_).size() + 1;
    for (size_t i = 0; i < nodeCount; ++i) {
        double nodeResult = 0;
        for (size_t j = 1; j < rowSize; ++j) {
            nodeResult += properties_[i * rowSize + j] * (*lastInputs_)[j - 1];
        }
        nodeResult += properties_[i * rowSize];;
        lastOutputs_[i] = elu_(nodeResult, eluAlpha_);
    }
}

std::vector<double>*
Layer::returnOut()
{
    return &lastOutputs_;
}

void
Layer::backpropagate(const double learningRate)
{
    assert(downstreamGradientsSource_ != NULL);
    assert(lastInputs_ != NULL);

    const std::vector<double>& downstreamGradients = *downstreamGradientsSource_;
    const size_t nodeCount = lastOutputs_.size();
    const size_t inputCount = (*lastInputs_).size();
    const size_t rowSize = inputCount + 1;

    assert(downstreamGradients.size() == nodeCount);
    for (size_t i = 0; i < nodeCount; ++i) {
        const double activationDerivative = eluDerivative_(lastZs_[i], eluDerivativeAlpha_);
        double rawDelta = downstreamGradients[i] * activationDerivative;
        
        if (rawDelta > 5.0)  rawDelta = 5.0;
        if (rawDelta < -5.0) rawDelta = -5.0;
        deltas_[i] = rawDelta;
    }
    for (size_t j = 0; j < inputCount; ++j) {
        backGradients_[j] = 0.0;
        for (size_t i = 0; i < nodeCount; ++i) {
            backGradients_[j] += deltas_[i] * properties_[i * rowSize + (j + 1)];
        }
    }
    for (size_t i = 0; i < nodeCount; ++i) {
        properties_[i * rowSize] -= learningRate * deltas_[i];
        for (size_t j = 1; j < rowSize; ++j) {
            properties_[i * rowSize + j] -= learningRate * deltas_[i] * (*lastInputs_)[j - 1];
        }
    }
}
void
Layer::setInputSource(std::vector<double>* source) {
    lastInputs_ = source;
}

void
Layer::setGradientSource(const std::vector<double>* source) {
    downstreamGradientsSource_ = source;
}

std::vector<double>*
Layer::returnGrad() {
    return &backGradients_;
}


#endif /// __LAYER_CPP__
