import numpy as np
import numpy.random as nprand


def seqsToMatrix(x_in):
    return np.array(x_in)


def splitFeaturesTargets(xy_in):
    features, targets = zip(*xy_in)
    return seqsToMatrix(features), seqsToMatrix(targets)


class CostFunction(object):
    def cost(self, out, targets):
        pass

    def derivative(self, out, targets):
        pass


class MeanSquaredError(CostFunction):
    def cost(self, out, targets):
        sq_error = np.square(out - targets)
        return .5 * sq_error.mean(0).sum()

    def derivative(self, out, targets):
        return out - targets


class GradientModifier(object):
    def get_learning_step(self, gradient):
        pass


class SimpleLearner(GradientModifier):
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate

    def get_learning_step(self, gradient):
        return self.learning_rate * gradient


class AdaDelta(GradientModifier):
    def __init__(self, decay_rate=0.95, eps=1e-6):
        self.decay_rate = decay_rate
        self.eps = eps
        self.gradAccum = None
        self.upAccum = None

    def get_learning_step(self, gradient):
        if self.gradAccum is None or self.upAccum is None:
            self.gradAccum = np.zeros(gradient.shape)
            self.upAccum = np.zeros(gradient.shape)
        #from http://arxiv.org/pdf/1212.5701v1.pdf
        self.gradAccum = self.decay_rate * self.gradAccum + (1 - self.decay_rate) * (gradient ** 2.0)
        rms_g = np.sqrt(self.gradAccum + self.eps)
        rms_delta = np.sqrt(self.upAccum + self.eps)
        update = (rms_delta / rms_g) * gradient
        self.upAccum = (self.decay_rate * self.upAccum) + (1-self.decay_rate) * (update ** 2.0)
        return update


class RMSProp(GradientModifier):
    def __init__(self, learning_rate=0.0001,
                 decay_rate=0.9,
                 smoothing_value=1e-6):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.smoothing_value = smoothing_value
        self.mean_square = None


    def get_learning_step(self, gradient):
        if self.mean_square is None:
            self.mean_square = np.zeros(gradient.shape)
        self.mean_square = self.decay_rate * self.mean_square + (1.0 - self.decay_rate) * gradient ** 2.0
        root_mean_square = np.sqrt(self.mean_square)
        smoothed = root_mean_square + self.smoothing_value
        return self.learning_rate * gradient / smoothed

class Layer(object):
    def __init__(self, name, input_size, output_size, act_func, grad_modifier=None):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.act_func = act_func
        if grad_modifier is None:
            self.grad_modifier = SimpleLearner(0.001)
        else:
            self.grad_modifier = grad_modifier
        self.last_inputs = None
        self.last_derivatives = None
        self.next_errors = None
        self.weights = nprand.uniform(-0.05, 0.05, (input_size + 1, output_size))
        self.output = None

    def forward_prop(self, x_in, w=None):
        if w is None:
            w = self.weights
        self.last_inputs = np.hstack((np.ones((x_in.shape[0], 1)), x_in))
        z = np.dot(self.last_inputs, w)
        self.last_derivatives = self.act_func.derivative(z)
        self.output = self.act_func.activation(z)
        return self.output

    def calculate_derivative(self, errors, w=None):
        if w is None:
            w = self.weights
        num_examples = errors.shape[0]
        little_delta = self.last_derivatives * errors
        drop_biases = w[1:, :]
        self.next_errors = np.dot(little_delta, drop_biases.transpose())
        derivative_sums = np.dot(self.last_inputs.transpose(), little_delta)
        derivative_avgs = derivative_sums / num_examples
        return derivative_avgs

    def back_prop(self, errors,
                  reg_lambda, w=None):
        if w is None:
            w = self.weights
        num_examples = errors.shape[0]
        derivative_avgs = self.calculate_derivative(errors)
        regularization = (reg_lambda / num_examples) * w
        regged_derivative = derivative_avgs + regularization
        learning_step = self.grad_modifier.get_learning_step(regged_derivative)
        w -= learning_step


class NeuralNet(object):

    def __init__(self, layers, cost_func, reg_lambda=0.0):
        self.layers = layers
        self.cost_func = cost_func
        self.reg_lambda = reg_lambda

    def forward_prop(self, x_in):
        processed = None
        for l in self.layers:
            if not processed:
                l.forward_prop(x_in)
            else:
                l.forward_prop(processed.output)
            processed = l
        return processed.output

    def back_prop(self, x_in, targets):
        output = self.forward_prop(x_in)
        error = self.cost_func.derivative(output, targets)
        processed = None
        for l in reversed(self.layers):
            if not processed:
                l.back_prop(error, self.reg_lambda)
            else:
                l.back_prop(processed.nextErrors, self.reg_lambda)
            processed = l

    def get_gradients(self, x_in, targets):
        output = self.forward_prop(x_in)
        error = self.cost_func.derivative(output, targets)
        processed = None
        grads = []
        for l in reversed(self.layers):
            if not processed:
                deriv = l.calculate_derivative(error)
            else:
                deriv = l.calculate_derivative(processed.next_errors)
            processed = l
            grads.append(deriv)
        grads.reverse()
        return grads

    def train(self, x_in, epochs, batch_size=32):
        shuffled = x_in
        np.random.shuffle(shuffled)
        validation_size = int(len(shuffled) * 0.2)
        validation_x, validation_y = splitFeaturesTargets(shuffled[:validation_size])
        training = shuffled[validation_size:]

        last_cost = float("inf")
        cur_cost = self.cost_func.cost(self.forward_prop(validation_x), validation_y)
        cur_epoch = 0

        while last_cost > cur_cost and cur_epoch < epochs:
            train_shuffle = training
            np.random.shuffle(train_shuffle)
            batches = [train_shuffle[i: i+batch_size] for i in xrange(0, len(train_shuffle), batch_size)]
            for matrix in batches:
                train_x, train_y = splitFeaturesTargets(matrix)
                self.back_prop(train_x, train_y)
            last_cost = cur_cost
            cur_cost = self.cost_func.cost(self.forward_prop(validation_x), validation_y)
            print "epoch: " + str(cur_epoch)
            print "cost:" + str(cur_cost)
            cur_epoch += 1

    def numerical_gradient(self, x_in, targets, epsilon=1e-5):
        grads = []
        for l in self.layers:
            m, n = l.weights.shape
            out = np.zeros((m, n))
            for i in xrange(0, m):
                for j in xrange(0, n):
                    temp = l.weights[i, j]
                    l.weights[i, j] = temp - epsilon
                    loss1 = self.cost_func.cost(self.forward_prop(x_in), targets)
                    l.weights[i, j] = temp + epsilon
                    loss2 = self.cost_func.cost(self.forward_prop(x_in), targets)
                    l.weights[i, j] = temp  # set back to what it was before doing next
                    out[i, j] = (loss2 - loss1) / (2.0 * epsilon)
            grads.append(out)
        return grads
