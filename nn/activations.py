__author__ = 'joseph'

import numpy as np


class ActivationFunction(object):
    def activation(self, x):
        pass

    def derivative(self, x):
        pass


class LinearActivation(ActivationFunction):
    def activation(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sig_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


class SigmoidActivation(ActivationFunction):
    def activation(self, x):
        return sigmoid(x)

    def derivative(self, x):
        return sig_deriv(x)


def relu(x):
    return x * (x > 0.0)


def relu_deriv(x):
    return x > 0.0


class ReluActivation(ActivationFunction):
    def activation(self, x):
        return relu(x)

    def derivative(self, x):
        return relu_deriv(x)


def leaky_relu(x):
    return (x * (x > 0.0)) + (0.01 * x * (x <= 0.0))


def leaky_relu_deriv(x):
    return (0.99 * (x > 0.0)) + 0.01


class LeakyReluActivation(ActivationFunction):
    def activation(self, x):
        return leaky_relu(x)

    def derivative(self, x):
        return leaky_relu_deriv(x)


def tanh_deriv(x):
    tanh_squared = np.tanh(x) ** 2.0
    return 1.0 - tanh_squared


class TanhActivation(ActivationFunction):
    def activation(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return tanh_deriv(x)
