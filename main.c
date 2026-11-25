
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neural_network.h"

#define SAMPLES 4
#define EPOCHS 10000
#define LEARNING_RATE 0.1

int main(void)
{
    srand((unsigned int)time(NULL));

    // AND gate dataset
    double X[SAMPLES][NN_INPUTS] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double Y[SAMPLES] = {0, 0, 0, 1};

    NN_Neuron network;

    // Initialize network
    nn_init(&network);

    // Select loss function
    NN_Loss loss_type = NN_BCE;

    FILE *file = fopen("loss.csv", "w");

    if (file == NULL) {
        printf("Cannot create loss.csv\n");
        return 1;
    }

    fprintf(file, "epoch,loss\n");

    // Training
    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        double loss = nn_train_epoch(
            &network,
            X,
            Y,
            SAMPLES,
            loss_type,
            LEARNING_RATE
        );

        fprintf(file, "%d,%.10f\n",
                epoch + 1, loss);

        if (epoch % 100 == 0) {
            printf("Epoch %d | Loss %.6f\n",
                   epoch + 1, loss);
        }
    }

    fclose(file);

    // Prediction
    printf("\nPredictions:\n");

    for (int i = 0; i < SAMPLES; i++) {

        double output = nn_predict(&network, X[i]);

        printf("%.0f AND %.0f = %.4f\n",
               X[i][0], X[i][1], output);
    }

    return 0;
}
