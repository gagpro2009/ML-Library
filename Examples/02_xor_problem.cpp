#include "../Lib/Net.hpp"
#include <iostream>
#include <cstdlib>

int
main()
{
    std::vector<size_t> topology;
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);

    std::vector<double(*)(double,double)> funcs;
    funcs.push_back(Activation::leakyRelu);
    funcs.push_back(Activation::sigmoid);

    std::vector<double(*)(double,double)> derivs;
    derivs.push_back(Activation::leakyReluDerivative);
    derivs.push_back(Activation::sigmoidDerivative);

    std::vector<double> alphas(2, 0.01);

    Network net(topology, funcs, derivs, alphas, alphas, Init::heUniform, Loss::bce, 0.01);

    double inputsData[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    double targetsData[4][1] = { {0},    {1},    {1},    {0}    };

    std::vector<double> in(2);
    std::vector<double> target(1);

    std::cout << "--- Training Non-Linear XOR Gate ---\n";
    for (long epoch = 0; epoch < 500000; ++epoch) {
        int idx = std::rand() % 4;
        in[0] = inputsData[idx][0];
        in[1] = inputsData[idx][1];
        target[0] = targetsData[idx][0];

        net.train(in, target);
    }

    for (int i = 0; i < 4; ++i) {
        in[0] = inputsData[i][0];
        in[1] = inputsData[i][1];
        const std::vector<double>& pred = net.run(in);
        std::cout << "In: [" << in[0] << ", " << in[1] << "]"
                  << " -> Pred: " << pred[0]
                  << " (Class: " << (pred[0] > 0.5 ? 1 : 0) << ")"
                  << " | Expected: " << targetsData[i][0] << "\n";
    }

    return 0;
}
