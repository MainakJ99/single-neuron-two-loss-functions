
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define NN_INPUTS 2

typedef enum {
    NN_MSE,
    NN_BCE
} NN_Loss;

// Neural network structure
typedef struct {
    double weights[NN_INPUTS];
    double bias;
} NN_Neuron;

// Initialize network
void nn_init(NN_Neuron *nn);

// Predict output
double nn_predict(const NN_Neuron *nn,
                  const double input[]);

// Calculate loss
double nn_loss(double target,
               double prediction,
               NN_Loss loss_type);

// Train using one sample
void nn_train_sample(NN_Neuron *nn,
                     const double input[],
                     double target,
                     NN_Loss loss_type,
                     double learning_rate);

// Train for one complete epoch
double nn_train_epoch(NN_Neuron *nn,
                      const double X[][NN_INPUTS],
                      const double Y[],
                      int samples,
                      NN_Loss loss_type,
                      double learning_rate);

#endif
