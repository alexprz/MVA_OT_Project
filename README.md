# Stuy of particle gradient flow

## Code organisation

All the experiments presented in the article can be reproduced using the
`XP*-*.py` files.

### The sparse deconvolution example:

* `XP1-1_sd1_dirichlet.py` reproduces the study of the dirichlet kernel in sparse deconvolution.
* `XP1-2_sd1_gaussian.py` reproduces the study of the gaussian kernel in sparse deconvolution.
* `XP1-3_sd1_lambda.py` reproduces the study of the lambda in sparse deconvolution.
* `XP1-4_sd1_initialization.py` reproduces the study of initialization in sparse deconvolution.

### The two-layer neural network example:

* `XP2-1_tln_relu_squared.py` reproduces the study of the relu activation and squared loss in the two-layer network example
* `XP2-2_tln_lambda.py` reproduces the study of lambda in the two-layer network example
* `XP2-3_tln_initialization.py` reproduces the study of initialization in the two-layer network example
* `XP2-4_tln_relu_logistic.py` reproduces the study of the relu activation and logistic loss in the two-layer network example

### Core classes:
The core classes for each of the two examples are implemented in:
* `sparse_deconvolution_1D.py` for the sparse deconvolution example.
* `two_layer_nn.py` for the two-layer neural network example.

The forward-backward algorithm and the stochastic gradient descent algorithm are implemented in
* `optimizer.py`.

### Miscelaneous:
Other files implement miscelanious functions and classes:
* `activations.py` the activation function classes.
* `kernels.py` the kernel classes.
* `losses.py` the loss function classes.
* `parameters.py` the classes structuring the parameters of each example, the common parameters and the custom parameters for each experiment.
* `plot.py` the functions used to plot the results.
* `tests.py` some tests.
