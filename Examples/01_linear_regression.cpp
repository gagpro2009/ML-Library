#include "../Lib/Net.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>

int
main()
{
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    std::vector<size_t> topology;
    topology.push_back(1);
    topology.push_back(1);

    std::vector<double(*)(double,double)> funcs(1, Activation::raw);
    std::vector<double(*)(double,double)> derivs(1, Activation::rawDerivative);
    std::vector<double> alphas(1, 1.0);

    Network net(topology, funcs, derivs, alphas, alphas, Init::ones, Loss::mse, 0.0001);

    std::vector<double> in(1);
    std::vector<double> target(1);

    std::cout << "--- Training Linear Regression: f(x) = 2x ---\n";
    for (long epoch = 0; epoch < 1000000; ++epoch) {
        in[0] = ((double)(std::rand() % 500) + 1.0) / 100.0;
        target[0] = in[0] * 2.0;
        net.train(in, target);
    }

    double testValues[] = {4.0, 5.0, 10.0, 15.0};
    for (size_t i = 0; i < 4; ++i) {
        in[0] = testValues[i];
        const std::vector<double>& pred = net.run(in);
        std::cout << "Input: " << in[0] 
                  << " | Predicted: " << pred[0] 
                  << " | Expected: " << testValues[i] * 2.0 << "\n";
    }
    return 0;
}
