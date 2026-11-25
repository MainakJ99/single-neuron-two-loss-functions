# Single Neuron with Two Loss Functions in C

## Overview

This project implements a **single artificial neuron in C** and trains it using two different loss functions:

- Mean Squared Error (MSE)
- Binary Cross-Entropy (BCE)

The purpose of this project is to understand how different loss functions affect the training of a single neuron.

During training, the loss values for both methods are recorded in a CSV file for further analysis.

## Features

- Implemented entirely in C
- Single neuron with two input features
- Sigmoid activation function
- MSE and BCE loss functions
- Gradient descent training
- Training loss recorded in a CSV file
- Simple AND gate dataset

## Project Structure

```text
single-neuron-two-loss-functions/
│
├── main.c
├── loss.csv
├── README.md
└── .gitignore
```

`loss.csv` is generated automatically when the program runs.

## Dataset

The neuron is trained on the AND gate truth table:

| Input 1 | Input 2 | Target |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

## Loss Functions

### Mean Squared Error (MSE)

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
\]

MSE measures the squared difference between the target and predicted output.

### Binary Cross-Entropy (BCE)

\[
BCE=-\frac{1}{N}\sum_{i=1}^{N}
[y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)]
\]

BCE is commonly used for binary classification problems.

## CSV Output

After training, the program writes the loss values to:

```text
loss_comparison.csv
```

The CSV file contains the training epoch and the corresponding loss values.

Example:

```csv
Epoch,MSE_Loss,BCE_Loss
0,0.2451,0.6932
1,0.2314,0.6528
2,0.2187,0.6145
```

The values above are illustrative. Actual values depend on the initialization and training parameters.

## Requirements

- GCC compiler
- C programming environment

## Compile and Run

Compile the program:

```bash
gcc main.c -o neuron.exe -lm
```

Run it in Windows PowerShell:

```powershell
.\neuron.exe
```

The program trains the neuron and generates the CSV file.

## Learning Objectives

This project demonstrates:

- How a single artificial neuron works
- How weights and bias are updated
- How sigmoid activation is used
- How MSE and BCE calculate prediction error
- How gradient descent trains a neuron
- How C programs can save training results to CSV files

## Future Improvements

- Test different learning rates
- Experiment with different weight initializations
- Add more datasets
- Compare convergence over different numbers of epochs
- Extend the single neuron into a multilayer neural network

## License

This project is open-source and available under the MIT License.
