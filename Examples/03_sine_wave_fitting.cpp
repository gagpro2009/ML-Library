#include "../Lib/Net.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

int
main()
{
    std::vector<size_t> topology;
    topology.push_back(1);
    topology.push_back(8);
    topology.push_back(8);
    topology.push_back(1);

    std::vector<double(*)(double,double)> funcs(3, Activation::elu);
    std::vector<double(*)(double,double)> derivs(3, Activation::eluDerivative);
    std::vector<double> alphas(3, 1.0);

    Network net(topology, funcs, derivs, alphas, alphas, Init::heNormal, Loss::huber, 0.001);

    std::vector<double> in(1);
    std::vector<double> target(1);

    std::cout << "--- Training Sine Curve Fitting [0, PI] ---\n";
    for (long epoch = 0; epoch < 1000000; ++epoch) {
        double x = ((double)std::rand() / RAND_MAX) * 3.14159265;
        in[0] = x;
        target[0] = std::sin(x);

        net.train(in, target);
    }

    for (double x = 0.0; x <= 3.14159; x += 0.5) {
        in[0] = x;
        const std::vector<double>& pred = net.run(in);
        std::cout << "x: " << x 
                  << " | sin(x): " << std::sin(x) 
                  << " | Pred: " << pred[0] << "\n";
    }

    return 0;
}
