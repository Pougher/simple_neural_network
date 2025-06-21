#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>

struct NeuralNet {
    std::vector<std::vector<float>> weights;

    int input_nodes;
    float learning_rate;

    // initializes the perceptron with the correct number of input nodes and
    // the learning rate
    NeuralNet(int, float);

    // predict an output from a set of input nodes
    float predict(std::vector<float> &inputs);

    // acitvation function for the neural net
    float activation(float input);

    // trains the perceptron for the specified number of iterations
    void train(int);

    // generates a set of training data to be used in each iteration
    std::vector<float> generate_training_vector(float*);

    // tests for average error (runs a bunch of tests, and calculates the
    // mean error)
    float mean_error();
};

NeuralNet::NeuralNet(int input_nodes, float rate) : input_nodes(input_nodes),
    learning_rate(rate) {
    for (int i = 0; i < input_nodes + 1; i++) {
        std::vector<float> w;
        for (int k = 0; k < input_nodes; k++) {
            const float random_number =
                -1 + static_cast<float>(rand()) /
                (static_cast<float>(RAND_MAX/(2)));
            w.push_back(random_number);
        }

        weights.push_back(w);
    }
}

float NeuralNet::activation(float input) {
    // simple activation function: step
    // If the value is negative, predict -1.
    // If the value is positive, predict 1

    if (input < 0) return -1;
    if (input >= 0) return 1;

    return -1;
}

float NeuralNet::predict(std::vector<float> &inputs) {
    std::vector<float> node_results;
    for (int i = 0; i < input_nodes + 1; i++) {
        float sum = 0;
        for (int k = 0; k < input_nodes; k++) {
            sum += inputs[k] * weights[i][k];
        }

        node_results.push_back(activation(sum));
    }
    return node_results[input_nodes];
}

std::vector<float> NeuralNet::generate_training_vector(float *expected) {
    std::vector<float> training_data;

    int num_negative = 0;
    for (int i = 0; i < input_nodes; i++) {
        const float random_number = -1 + static_cast<float>(rand()) /
            (static_cast<float>(RAND_MAX/(2)));
        training_data.push_back(random_number);

        if (random_number < 0) {
            num_negative ++;
        }
    }

    *expected = (num_negative > (input_nodes / 2)) ? -1 : 1;
    return training_data;
}

void NeuralNet::train(int iterations) {
    for (int i = 0; i < iterations; i++) {
        float expected;
        std::vector training_data = generate_training_vector(&expected);
        float guess = predict(training_data);
        float error = expected - guess;

        //std::cout << "error: " << error << std::endl;
        if (error != 0) {
            for (int i = 0; i < input_nodes + 1; i++) {
                for (int k = 0; k < input_nodes; k++) {
                    weights[i][k] += learning_rate * error * training_data[k];
                }
            }
        }
    }
}

float NeuralNet::mean_error() {
    const int iterations = 100000;

    float accumulated_error = 0;

    for (int i = 0; i < iterations; i++) {
        float expected;
        std::vector training_data = generate_training_vector(&expected);
        float guess = predict(training_data);

        // take the absolute value, as error can be both positive and negative
        accumulated_error += fabs(expected - guess);
    }

    return (accumulated_error / static_cast<float>(iterations));
}

int main() {
    // My objective with this program is to return the most common sign in a
    // list of five integers. For example, the list (0.1, 0.4, -0.2, 1.0, 0.4)
    // should return a value of 1, since the most common sign is +1.
    //
    // My implementation is a simple neural network, with 1 hidden layer, 5
    // input layers and an output layer, with a simple activation function and
    // no bias nodes.
    //
    // Due to my simplistic implementation, the average error rate after 10
    // million training iterations was ~0.3, which is often a drastic
    // improvement over the initial error rate (usually around 1-1.5)
    // 10 million training iterations seems a bit much though, as you quickly
    // see diminishing returns after roughly 1 million iterations in my
    // observations.

    srand(time(NULL));

    // I have tried a bunch of different learning rates, and this one seems
    // to work best for this purpose
    NeuralNet network(5, 0.000001);

    std::vector<float> test_values{0.2, 0.3, 0.4, -0.2, 0.3};

    std::cout << "Mean error before training: " << network.mean_error()
        << std::endl;
    network.train(10000000);

    std::cout << "Mean error after training: " << network.mean_error()
        << std::endl;

    return 0;
}
