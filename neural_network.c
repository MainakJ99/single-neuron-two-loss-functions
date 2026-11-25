
#include "neural_network.h"

#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-12

static double sigmoid(double z)
{
    if (z < -500.0) return 0.0;
    if (z > 500.0) return 1.0;

    return 1.0 / (1.0 + exp(-z));
}

void nn_init(NN_Neuron *nn)
{
    for (int i = 0; i < NN_INPUTS; i++) {
        nn->weights[i] =
            2.0 * ((double)rand() / RAND_MAX) - 1.0;
    }

    nn->bias =
        2.0 * ((double)rand() / RAND_MAX) - 1.0;
}

double nn_predict(const NN_Neuron *nn,
                 const double input[])
{
    double z = nn->bias;

    for (int i = 0; i < NN_INPUTS; i++) {
        z += nn->weights[i] * input[i];
    }

    return sigmoid(z);
}

double nn_loss(double target,
               double prediction,
               NN_Loss loss_type)
{
    if (loss_type == NN_MSE) {
        double error = prediction - target;
        return error * error;
    }

    if (prediction < EPSILON)
        prediction = EPSILON;

    if (prediction > 1.0 - EPSILON)
        prediction = 1.0 - EPSILON;

    return -(
        target * log(prediction) +
        (1.0 - target) * log(1.0 - prediction)
    );
}

void nn_train_sample(NN_Neuron *nn,
                     const double input[],
                     double target,
                     NN_Loss loss_type,
                     double learning_rate)
{
    // Forward propagation
    double prediction = nn_predict(nn, input);

    // Calculate output delta
    double delta;

    if (loss_type == NN_BCE) {
        // Sigmoid + BCE
        delta = prediction - target;
    } else {
        // MSE + Sigmoid
        delta = 2.0 *
                (prediction - target) *
                prediction *
                (1.0 - prediction);
    }

    // Update weights
    for (int i = 0; i < NN_INPUTS; i++) {
        double gradient = delta * input[i];

        nn->weights[i] -=
            learning_rate * gradient;
    }

    // Update bias
    nn->bias -= learning_rate * delta;
}

double nn_train_epoch(NN_Neuron *nn,
                      const double X[][NN_INPUTS],
                      const double Y[],
                      int samples,
                      NN_Loss loss_type,
                      double learning_rate)
{
    double total_loss = 0.0;

    for (int sample = 0; sample < samples; sample++) {

        // Calculate loss before updating
        double prediction = nn_predict(nn, X[sample]);

        total_loss += nn_loss(
            Y[sample],
            prediction,
            loss_type
        );

        // Train this sample
        nn_train_sample(
            nn,
            X[sample],
            Y[sample],
            loss_type,
            learning_rate
        );
    }

    return total_loss / samples;
}
