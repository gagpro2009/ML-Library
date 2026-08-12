# C++ Neural Network Library

A lightweight, zero-allocation C++ neural network library designed for high performance, modular activation & loss functions, customizable layer initializations, and legacy C++98/03 portability.

---

## Key Features

- **Zero-Allocation Passes**: All internal memory (intermediate outputs, layer gradients, deltas) is pre-allocated upon network construction. `run()` and `train()` execute without dynamic heap allocations during execution loops.
- **Gradient Clipping**: Delta gradients are clamped to `[-5.0, 5.0]` during backpropagation to prevent exploding gradients.
- **Per-Layer Activation Control**: Custom activation functions and derivatives can be configured independently for each layer.
- **C++98 / C++03 Compatibility**: Built using standard C++ structures without modern library overhead, optimized for high performance and backward portability.
- **Comprehensive Loss & Init Utilities**: Includes MSE, MAE, Huber, Log-Cosh, BCE, Categorical Cross-Entropy, Hinge, and Cosine Proximity loss functions, along with He, Xavier, Uniform, Zero, and Ones weight initializers.

---

## Directory Structure

```text
.
├── Examples
│   ├── 01_linear_regression.cpp     # Single-neuron linear model fitting f(x) = 2x
│   ├── 02_xor_problem.cpp           # Multi-layer perceptron solving non-linear XOR
│   ├── 03_sine_wave_fitting.cpp     # Continuous function approximation on [0, PI]
│   └── 04_zero_allocation_benchmark.cpp # Performance and throughput benchmark
├── Lib
│   ├── headers
│   │   ├── Layer.hpp                # Layer class declaration
│   │   ├── Network.hpp              # Network class declaration
│   │   └── Utils.hpp                # Declarations for Loss, Activation, and Init
│   ├── Net.hpp                      # Main wrapper header (includes headers & templates)
│   └── templates
│       ├── Layer.cpp                # Layer backpropagation & feedforward algorithms
│       ├── Network.cpp              # Network execution & training loop implementation
│       └── Utils.cpp                # Implementations of loss, activation, and weight initializers
```

---

## Setting Up and Using the `Network` Class

The core container class is `Network`. It links adjacent layers via raw pointers to allow zero-allocation forward and backward passes.

### Constructor Signature

```cpp
Network(const std::vector<size_t>& topology,
        const std::vector<double(*)(double,double)>& elus,
        const std::vector<double(*)(double,double)>& eluDerivatives,
        const std::vector<double>& eluAlphas,
        const std::vector<double>& eluDerivativeAlphas,
        void(*init)(std::vector<double>&, size_t, size_t),
        void(*loss)(const std::vector<double>&, const std::vector<double>&, std::vector<double>&),
        const double learnRate);
```

### Parameter-by-Parameter Breakdown

#### 1. `topology` (`const std::vector<size_t>&`)
Specifies the architecture (number of neurons per layer):
- `topology[0]`: Input layer dimension.
- `topology[1]` to `topology[N-1]`: Hidden layer dimensions.
- `topology.back()`: Output layer dimension.
- Total number of trainable weight layers $N = \text{topology.size()} - 1$.

#### 2. `elus` (`const std::vector<double(*)(double,double)>&`)
A vector of function pointers defining activation functions for each trainable layer.
- Size must equal $N$ (`topology.size() - 1`).
- Function signature requirement: `double func(double x, double alpha)`.
- Available options in the `Activation::` namespace:
  - `Activation::raw` (Identity/Linear)
  - `Activation::relu`
  - `Activation::leakyRelu`
  - `Activation::relu6`
  - `Activation::elu`
  - `Activation::sigmoid`

#### 3. `eluDerivatives` (`const std::vector<double(*)(double,double)>&`)
A vector of function pointers for the activation derivative functions corresponding to each entry in `elus`.
- Size must equal $N$ (`topology.size() - 1`).
- Function signature requirement: `double deriv(double x, double alpha)`.
- Options in `Activation::`: `rawDerivative`, `reluDerivative`, `leakyReluDerivative`, `relu6Derivative`, `eluDerivative`, `sigmoidDerivative`.

#### 4. `eluAlphas` (`const std::vector<double>&`)
Alpha scaling/slope parameters passed as the second argument to each layer's activation function.
- Size must equal $N$ (`topology.size() - 1`).
- Used by parametric functions (e.g., negative slope for `leakyRelu` like `0.01`, capping threshold for `relu6`, or alpha scalar for `elu`).
- For standard activations (`relu`, `sigmoid`, `raw`), pass `1.0`.

#### 5. `eluDerivativeAlphas` (`const std::vector<double>&`)
Alpha parameters passed into the activation derivative functions.
- Size must equal $N$ (`topology.size() - 1`).
- Typically identical to `eluAlphas`.

#### 6. `init` (`void(*)(std::vector<double>&, size_t, size_t)`)
Global weight and bias initialization routine applied across all layers during network construction.
- Function signature: `void init(std::vector<double>& weights, size_t fanIn, size_t fanOut)`.
- Options available in the `Init::` namespace:
  - `Init::heUniform`: He uniform distribution ($\\sqrt{6 / \\text{fanIn}}$)
  - `Init::heNormal`: He normal distribution via Box-Muller transform ($\\sigma = \\sqrt{2 / \\text{fanIn}}$)
  - `Init::xavierUniform`: Glorot/Xavier uniform ($\\sqrt{6 / (\\text{fanIn} + \\text{fanOut})}$)
  - `Init::randomUniform`: Uniform distribution in `[-1.0, 1.0]`
  - `Init::ones`: Constant `1.0` initialization
  - `Init::zero`: Constant `0.0` initialization

#### 7. `loss` (`void(*)(const std::vector<double>&, const std::vector<double>&, std::vector<double>&)`)
The cost/loss function used to compute loss gradients at the output layer during backpropagation.
- Function signature: `void loss(const std::vector<double>& pred, const std::vector<double>& expect, std::vector<double>& grad)`.
- Options available in the `Loss::` namespace:
  - `Loss::mse`: Mean Squared Error
  - `Loss::bce`: Binary Cross-Entropy (clamped at $\\epsilon = 10^{-15}$)
  - `Loss::huber`: Huber Loss (robust to outliers, delta = 1.0)
  - `Loss::mae`: Mean Absolute Error
  - `Loss::logCosh`: Logarithm of Hyperbolic Cosine
  - `Loss::relativeMse`: Relative Mean Squared Error
  - `Loss::categoricalCrossEntropy`: Multiclass Cross-Entropy
  - `Loss::hinge`: Max-margin / SVM Loss
  - `Loss::cosineProximity`: Cosine Similarity Loss

#### 8. `learnRate` (`const double`)
Learning rate scalar ($\\eta$) applied to weight and bias parameter updates during gradient descent.

---

## Member Functions

```cpp
// Training pass: executes forward pass, evaluates loss, and backpropagates gradients
void train(std::vector<double>& input, std::vector<double>& output);

// Inference / Prediction pass: zero-allocation evaluation, returns reference to output vector
const std::vector<double>& run(std::vector<double>& input);
```

---

## API Overview Tables

### 1. Activation Functions (`Activation::`)

| Activation | Function Pointer | Derivative Pointer | Description |
| :--- | :--- | :--- | :--- |
| **Linear / Raw** | `Activation::raw` | `Activation::rawDerivative` | Scaled identity function ($x \cdot \alpha$) |
| **ReLU** | `Activation::relu` | `Activation::reluDerivative` | $\\max(0, x)$ |
| **Leaky ReLU** | `Activation::leakyRelu` | `Activation::leakyReluDerivative` | $x > 0 ? x : \\alpha \cdot x$ |
| **ReLU6** | `Activation::relu6` | `Activation::relu6Derivative` | Capped ReLU at max value $6.0$ (or $\\alpha$) |
| **ELU** | `Activation::elu` | `Activation::eluDerivative` | Exponential Linear Unit |
| **Sigmoid** | `Activation::sigmoid` | `Activation::sigmoidDerivative` | Logistic Sigmoid function |

### 2. Weight Initialization (`Init::`)

- `Init::heUniform`: He uniform initialization ($\\sqrt{6 / \\text{fanIn}}$)
- `Init::heNormal`: He normal distribution ($\\sigma = \\sqrt{2 / \\text{fanIn}}$)
- `Init::xavierUniform`: Glorot/Xavier uniform distribution ($\\sqrt{6 / (\\text{fanIn} + \\text{fanOut})}$)
- `Init::randomUniform`: Uniform distribution in `[-1, 1]`
- `Init::zero`: Fills weights with $0.0$
- `Init::ones`: Fills weights with $1.0$

### 3. Loss Functions (`Loss::`)

- `Loss::mse`: Mean Squared Error
- `Loss::mae`: Mean Absolute Error
- `Loss::huber`: Huber Loss ($\\delta = 1.0$)
- `Loss::logCosh`: Logarithm of Hyperbolic Cosine
- `Loss::relativeMse`: Relative Mean Squared Error
- `Loss::bce`: Binary Cross-Entropy with numerical clamping ($\\epsilon = 10^{-15}$)
- `Loss::categoricalCrossEntropy`: Categorical Cross-Entropy
- `Loss::hinge`: Hinge / Max-Margin loss
- `Loss::cosineProximity`: Cosine similarity loss

---

## Code Examples

### 1. Linear Regression (`Examples/01_linear_regression.cpp`)

Fits a linear model $f(x) = 2x$ using identity activation (`Activation::raw`) and MSE loss (`Loss::mse`).

```cpp
#include "../Lib/Net.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
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
```

---

### 2. XOR Problem (`Examples/02_xor_problem.cpp`)

Solves the non-linear XOR gate using a 3-layer architecture ($2 \to 4 \to 1$), Leaky ReLU, Sigmoid, He Uniform initialization, and Binary Cross-Entropy loss.

```cpp
#include "../Lib/Net.hpp"
#include <iostream>
#include <cstdlib>

int main() {
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
```

---

### 3. Sine Wave Fitting (`Examples/03_sine_wave_fitting.cpp`)

Approximates continuous $\sin(x)$ over $[0, \pi]$ using a 4-layer network ($1 \to 8 \to 8 \to 1$), ELU activation, He Normal initialization, and Huber loss.

```cpp
#include "../Lib/Net.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

int main() {
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
```

---

### 4. Zero-Allocation Benchmark (`Examples/04_zero_allocation_benchmark.cpp`)

Measures training pass latency across 100,000 iterations on a dense $100 \to 200 \to 50 \to 10$ architecture.

```cpp
#include "../Lib/Net.hpp"
#include <iostream>
#include <ctime>

int main() {
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
```

---
