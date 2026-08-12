#include "../Lib/Net.hpp"
#include <iostream>
#include <ctime>

int
main()
{
    std::vector<size_t> topology;
    topology.push_back(100);
    topology.push_back(200);
    topology.push_back(50);
    topology.push_back(10);

    std::vector<double(*)(double,double)> funcs(3, Activation::relu);
    std::vector<double(*)(double,double)> derivs(3, Activation::reluDerivative);
    std::vector<double> alphas(3, 1.0);

    Network net(topology, funcs, derivs, alphas, alphas, Init::heUniform, Loss::mse, 0.001);

    std::vector<double> in(100, 1.0);
    std::vector<double> target(10, 0.5);

    const size_t iterations = 100000;
    std::cout << "--- Running Benchmark (" << iterations << " iterations) ---\n";

    std::clock_t startTime = std::clock();
    for (size_t i = 0; i < iterations; ++i) {
        net.train(in, target);
    }
    std::clock_t endTime = std::clock();

    double totalSecs = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
    double avgMs = (totalSecs / iterations) * 1000.0;

    std::cout << "Total Training Loop Time: " << totalSecs << " seconds\n";
    std::cout << "Average Train Pass Time: " << avgMs << " ms\n";

    return 0;
}
