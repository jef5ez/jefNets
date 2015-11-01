__author__ = 'jef5ez'

from collections import OrderedDict
import numpy as np
import numpy.random as npr
import sklearn.metrics as sklm
import sys
import theano
import theano.sandbox.neighbours as TSN
from theano.ifelse import ifelse
from theano import tensor as T
from theano.printing import Print as tPrint
from theano.tensor.shared_randomstreams import RandomStreams

import time

try:
    import cPickle as pickle
except:
    import pickle


def create_shared_matrix(row_dim, col_dim, name):
    shape = (row_dim, col_dim)
    return theano.shared(0.2 * np.random.uniform(-1.0, 1.0, shape).astype(theano.config.floatX),
                         name=name)

def create_shared_3d(i_dim, row_dim, col_dim, name, broadcast=(False, False, False)):
    shape = (i_dim, row_dim, col_dim)
    return theano.shared(0.2 * np.random.uniform(-1.0, 1.0, shape).astype(theano.config.floatX),
                         name=name,
                         broadcastable=broadcast)


def create_shared_4d(cur_feats, prev_feats, row_dim, col_dim, name):
    shape = (cur_feats, prev_feats, row_dim, col_dim)
    return theano.shared(0.2 * np.random.uniform(-1.0, 1.0, shape).astype(theano.config.floatX),
                         name=name)


def one_of_n(idx, size):
    zeros = np.zeros([size])
    zeros[idx] = 1
    return zeros


def k_of_n(idxs, size):
    zeros = np.zeros([size])
    zeros[idxs] = 1
    return zeros


def pad_sentence_batch(sents):
    """
    :param sents: lists of lists of idxs
    :return: matrix padded to largest sentence length with neg ones
    """
    def pad_sentence(sentence, max_size):
        return np.pad(sentence, (0, max_size - len(sentence)), 'constant',
                      constant_values=-1)
    max_sent_len = max(map(len, sents))
    return [pad_sentence(x, max_sent_len) for x in sents]


class DCNN(object):
    def __init__(self, word_vecs, folding_layers, filters_per_layer, filter_widths, train_set,
                 valid_set, num_classes=None, word_dim=None, hidden_dim=None, k_top=3, decay=0.95,
                 epsilon=1e-6, dropout=1, rand_state=npr.RandomState(5), theano_rand=RandomStreams(seed=5),
                 activate=T.tanh, last_activate=T.nnet.softmax):
        """

        :param word_vecs: rows of word vectors
        :param word_dim: width of word vectors
        :param hidden_dim: width of embedded space
        :param train_set: list of lists of ints (i.e. words)
        :param valid_set: list of lists of ints (i.e. words)
        :param num_classes: number of softmax classes
        :param folding_layers: list of 0s/1s for which layers are folded
        :param filters_per_layer: list of ints, how many filters to learn at each layer
        :param filter_widths: list of ints, width of each of the filters at each layer
                              (all filters at the same layer have the same width)
        :param k_top: the k-max pooling size for the topmost layer
        :param decay: ada delta decay rate
        :param epsilon: ada delta epsilon
        :return:
        """

        # sentence (d x s) matrix, ie s word vectors with d dimensions
        # weight_mat = (d x m), word_dim by m, the width of the convolution
        # c = (d x s+m-1), the wide/full convolution matrix of sentence and weights

        # after convolution can optionally fold word dimension

        # k-max pooling given a sequence of length greater than k take the k highest
        # values from the sequence without reordering the sequence
        # This is done at the top most convolution layer

        # dynamic k max pooling is using k that is a function of the sequence length
        # paper recommends k = max(k_top, ceil((L-l)/L * len(s)))
        # where L is the # of conv layers and l is the current layer

        # after the pooling is done bias is added and then it is activated
        self.rand_state = rand_state
        self.theano_rand = theano_rand
        self.word_vecs = theano.shared(word_vecs.astype(theano.config.floatX),
                                       name="wordVectors")
        self.train_set = train_set
        self.valid_set = valid_set
        if num_classes:
            self.num_classes = num_classes
        else:
            self.num_classes = len(train_set[0][1])
        if word_dim:
            self.word_dim = word_dim
        else:
            self.word_dim = word_vecs.shape[-1]
        self.k_top = k_top
        self.num_layers = 3
        dim_divisor = pow(2, sum(folding_layers))
        if self.word_dim % dim_divisor == 0:
            self.folding_layers = folding_layers
        else:
            raise Exception("Word dimension must be divisible by 2^(number of folded layers)")
        self.filter_heights = [self.word_dim]
        for i in self.folding_layers:
            if i:
                self.filter_heights += [self.filter_heights[-1]/2]
            else:
                self.filter_heights += [self.filter_heights[-1]]
        self.filters_per_layer = filters_per_layer
        self.filter_widths = filter_widths
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.word_dim
        self.theano_activate = activate
        self.unravelled_size = self.filter_heights[-1] * self.k_top * self.filters_per_layer[-1]

        self.filters = \
        [create_shared_4d(self.filters_per_layer[0], 1, self.word_dim, self.filter_widths[0],
                          "layer0")] + \
        [create_shared_4d(self.filters_per_layer[i], self.filters_per_layer[i - 1],
                          self.filter_heights[i], self.filter_widths[i], "layer"+str(i)) for i in
         xrange(1, self.num_layers)]
        # bias for each layer should be shape(num_filters, num_word_dim, 1)
        self.biases = [create_shared_3d(self.filters_per_layer[i], self.filter_heights[i + 1], 1,
                                        "layerbias" + str(i), (False, False, True)) for i in
                       xrange(self.num_layers)]
        self.hidden_layer = create_shared_matrix(self.unravelled_size, self.hidden_dim, 'hiddenWeights')
        self.theta = create_shared_matrix(self.hidden_dim, self.num_classes, 'classifyWeights')
        self.b = theano.shared(np.zeros((1, self.num_classes), dtype=theano.config.floatX),
                               broadcastable=(True, False),
                               name='softmaxBias')
        self.decay = decay
        self.epsilon = epsilon
        self.dropout = dropout

        def wide_convolve(inpt, filters):
            conv = T.nnet.conv2d(inpt, filters, border_mode='full')
            take = conv[:, :, :inpt.shape[2], :]
            t_shape = take.shape
            # final_shape = (t_shape[0], t_shape[1], inpt.shape[2], t_shape[3])
            extra_row = T.zeros((t_shape[0], t_shape[1], t_shape[2] + 1, t_shape[3]))
            padded = T.set_subtensor(extra_row[:, :, :-1, :], take)
            offset = T.set_subtensor(extra_row[:, :, 1:, :], take)
            diff = padded - offset
            final = diff[:, :, :-1, :]
            return final

        def fold(conv):
            c_shape = conv.shape
            pool_size = (1, conv.shape[-1])
            neighbors_to_pool = TSN.images2neibs(ten4=conv,
                                                 neib_shape=pool_size,
                                                 mode='ignore_borders')
            n_shape = neighbors_to_pool.shape
            paired = T.reshape(neighbors_to_pool, (n_shape[0] / 2, 2, n_shape[-1]))
            summed = T.sum(paired, axis=1)
            folded_out = T.reshape(summed, (c_shape[0], c_shape[1], c_shape[2] / 2, c_shape[3]),
                                   ndim=4)
            return folded_out

        def calculate_k(s_length, cur_layer):
            proportion = (self.num_layers - cur_layer) / self.num_layers
            return T.maximum(self.k_top, T.ceil(proportion * s_length))

        def k_max_pool(conv, k):
            c_shape = conv.shape
            # c_shape = tPrint('conv_shape')(c_shape)
            pool_size = (1, conv.shape[-1])
            neighbors_to_pool = TSN.images2neibs(ten4=conv,
                                                 neib_shape=pool_size,
                                                 mode='ignore_borders')
            arg_sorted = T.argsort(neighbors_to_pool, axis=1)
            top_k = arg_sorted[:, -k:]
            top_k_sorted = T.sort(top_k, axis=1)
            ii = T.repeat(T.arange(neighbors_to_pool.shape[0], dtype='int32'), k)
            jj = top_k_sorted.flatten()
            values = neighbors_to_pool[ii, jj]
            pooled_out = T.reshape(values, (c_shape[0], c_shape[1], c_shape[2], k), ndim=4)
            return pooled_out

        def bias_and_activate(pooled, bias):
            """

            :param pooled: 4d tensor, shape(num_sent, num_filts, num_word_dim, num_words)
            :param bias: 3d tensor, a bias vector for each filter with shape
                    (num_filters, num_word_dim, 1) last dimension broadcastable
            :return: active(pooled + bias)
            """
            return self.theano_activate(pooled + bias)

        def create_convolution_layer(inpt, filters, fold_bool, k, bias):
            conv = wide_convolve(inpt, filters)
            folded = ifelse(fold_bool, fold(conv), conv)
            pooled = k_max_pool(folded, k)
            return bias_and_activate(pooled, bias)

        def lookup_sentence(sentence):
            row_vecs = self.word_vecs[sentence]
            return T.transpose(row_vecs)

        def lookup_all(sentences):
            results, ups = theano.scan(lookup_sentence, sequences=[sentences])
            shape = results.shape
            return T.reshape(results, (shape[0], 1, shape[1], shape[2]), ndim=4)

        def conv_forward_prop_train(sentences):
            # layers = T.arange(self.num_layers)
            k = calculate_k(sentences.shape[-1], 1)
            # k = tPrint("first k")(k)
            sentences = T.switch(self.dropout,
                               theano_rand.binomial(sentences.shape,
                                                    p=0.8,
                                                    dtype=theano.config.floatX) * sentences,
                               sentences)
            first_layer = create_convolution_layer(sentences,
                                                   self.filters[0],
                                                   self.folding_layers[0],
                                                   k,
                                                   self.biases[0])
            k = calculate_k(sentences.shape[-1], 2)
            # k = tPrint("second k")(k)
            first_layer = T.switch(self.dropout,
                                 theano_rand.binomial(first_layer.shape,
                                                      dtype=theano.config.floatX) * first_layer,
                                 first_layer)
            second_layer = create_convolution_layer(first_layer,
                                                    self.filters[1],
                                                    self.folding_layers[1],
                                                    k,
                                                    self.biases[1])
            k = T.as_tensor(self.k_top)
            # k = tPrint("k_top")(k)
            second_layer = T.switch(self.dropout,
                                  theano_rand.binomial(second_layer.shape,
                                                       dtype=theano.config.floatX) * second_layer,
                                  second_layer)
            third_layer = create_convolution_layer(second_layer,
                                                   self.filters[2],
                                                   self.folding_layers[2],
                                                   k,
                                                   self.biases[2])
            third_layer = T.switch(self.dropout,
                                 theano_rand.binomial(third_layer.shape,
                                                      dtype=theano.config.floatX) * third_layer,
                                 third_layer)
            return third_layer

        def conv_forward_prop_test(sentences):
            # layers = T.arange(self.num_layers)
            k = calculate_k(sentences.shape[-1], 1)
            # k = tPrint("first k")(k)
            first_layer = create_convolution_layer(sentences,
                                                   T.switch(self.dropout,
                                                            self.filters[0] * 0.8,
                                                            self.filters[0]),
                                                   self.folding_layers[0],
                                                   k,
                                                   self.biases[0])
            k = calculate_k(sentences.shape[-1], 2)
            # k = tPrint("second k")(k)
            second_layer = create_convolution_layer(first_layer,
                                                    T.switch(self.dropout,
                                                            self.filters[1] * 0.5,
                                                            self.filters[1]),
                                                    self.folding_layers[1],
                                                    k,
                                                    self.biases[1])
            k = T.as_tensor(self.k_top)
            # k = tPrint("k_top")(k)
            third_layer = create_convolution_layer(second_layer,
                                                   T.switch(self.dropout,
                                                            self.filters[2] * 0.5,
                                                            self.filters[2]),
                                                   self.folding_layers[2],
                                                   k,
                                                   self.biases[2])
            return third_layer

        def embed(sentences):
            vectors = lookup_all(sentences)
            convolved = conv_forward_prop_train(vectors)
            flat = T.flatten(convolved, outdim=2)
            hidden = self.theano_activate(T.dot(flat, self.hidden_layer))
            return hidden

        def embed_test(sentences):
            vectors = lookup_all(sentences)
            convolved = conv_forward_prop_test(vectors)
            flat = T.flatten(convolved, outdim=2)
            h_weights = T.switch(self.dropout,
                                 self.hidden_layer * 0.5,
                                 self.hidden_layer)
            embedded= self.theano_activate(T.dot(flat, h_weights))
            return embedded

        def forward_prop(sentences):
            hidden = embed(sentences)
            classes = last_activate(T.dot(hidden, self.theta) + self.b)
            return classes

        def forward_prop_test(sentences):
            hidden = embed_test(sentences)
            classes = last_activate(T.dot(hidden, self.theta) + self.b)
            return classes

        self.params = self.filters + self.biases + [self.hidden_layer, self.theta, self.b,
                                                    self.word_vecs]

        def make_zero_shared_vars_like(a_shared, name_mod):
            zeros = np.zeros_like(a_shared.get_value())
            return theano.shared(zeros,
                                 name=a_shared.name + name_mod,
                                 broadcastable=a_shared.broadcastable)

        # holds expected values for ada delta
        self.param_grad_accums = map(lambda x: make_zero_shared_vars_like(x, "-grad"), self.params)
        self.param_up_accums = map(lambda x: make_zero_shared_vars_like(x, "-update"), self.params)

        s = T.imatrix("sentences")
        y = T.imatrix("response")
        beta = T.scalar("beta")

        def bootstrap_soft(beta, prediction, truth):
            return -1 * T.sum((beta * truth + (1-beta)*prediction)*T.log(prediction), axis=1).mean()

        def binary_bootstrap_soft(beta, prediction, truth):
            t = (beta * truth + (1-beta)*prediction)*T.log(prediction)
            f = (beta * (1-truth) + (1-beta)*(1-prediction)) * T.log(1 - prediction)
            return -1 * T.mean(t + f)

        def bootstrap_hard(beta, prediction, truth):
            zeros = T.zeros_like(prediction)
            am = T.argmax(prediction, axis=1)
            maxed = T.set_subtensor(zeros[T.arange(am.shape[0]), am], 1)
            return -1 * T.sum((beta * truth + (1-beta)*maxed)*T.log(prediction), axis=1).mean()

        # def binary_bootstrap_hard(beta, prediction, truth):
        #     t = (beta * truth + (1-beta)*prediction)*T.log(prediction)
        #     f = (beta * (1-truth) + (1-beta)*(1-prediction)) * T.log(1 - prediction)
        #     return -1 * T.mean(t + f)

        def multilabel_b_soft(beta, preds, ys):
            res, ups = theano.scan(lambda pred, truth: binary_bootstrap_soft(beta, pred, truth),
                                   sequences=[preds.T, ys.T])
            return T.sum(res)

        # def multilabel_b_hard(beta, preds, ys):
        #     zeros = T.zeros_like(preds)
        #     am = T.argmax(preds, axis=1)
        #     maxed = T.set_subtensor(zeros[T.arange(am.shape[0]), am], 1)
        #     res, ups = theano.scan(lambda pred, truth: bootstrap_hard(beta, pred, truth),
        #                            sequences=[preds.T, ys.T])
        #     return T.sum(res)

        def multilabel_cross_ent(preds, ys):
            res, ups = theano.scan(lambda p, t: T.nnet.binary_crossentropy(p, t).mean(),
                                   sequences=[preds.T, ys.T])
            return T.sum(res)

        self.individual_b_soft_sum = multilabel_b_soft(beta, forward_prop(s), y)

        # self.individual_b_hard_sum = multilabel_b_hard(beta, forward_prop(s), y)

        self.individual_cross_ent_sum = multilabel_cross_ent(forward_prop(s), y)

        self.bootstrap_soft_cost = bootstrap_soft(beta, forward_prop(s), y)

        self.bootstrap_hard_cost = bootstrap_hard(beta, forward_prop(s), y)

        self.squared_error = T.sum(T.mean((forward_prop(s)-y)**2, axis=0))

        self.cross_entropy = T.nnet.categorical_crossentropy(forward_prop(s), y).mean() #+ \
                    # T.sum(T.sqr(self.filters[0])) + T.sum(T.sqr(self.filters[1])) + \
                    # T.sum(T.sqr(self.filters[2])) +\
                    # T.sum(T.sqr(self.hidden_layer))
        b_soft_grads = T.grad(self.bootstrap_soft_cost, self.params)
        b_hard_grads = T.grad(self.bootstrap_hard_cost, self.params)
        sq_err_grads = T.grad(self.squared_error, self.params)
        cross_ent_grads = T.grad(self.cross_entropy, self.params)
        ind_b_soft_grads = T.grad(self.individual_b_soft_sum, self.params)
        # ind_b_hard_grads = T.grad(self.individual_b_hard_sum, self.params)
        ind_cross_ent_grads = T.grad(self.individual_cross_ent_sum, self.params)

        def ada_delta_step(next_gradient, decay_rate, eps, prev_grad, prev_up):
            # from http://arxiv.org/pdf/1212.5701v1.pdf
            grad_accum = (decay_rate * prev_grad) + (1-decay_rate) * (next_gradient ** 2)
            rms_g = T.sqrt(grad_accum + eps)
            rms_delta = T.sqrt(prev_up + eps)
            update = (rms_delta / rms_g) * next_gradient
            up_accum = (decay_rate * prev_up) + (1-decay_rate) * (update ** 2)
            return update, grad_accum, up_accum

        def create_update_tuple(param, grad, grad_acc, up_acc):
            # grad = tPrint("gradient for" + param.name)(grad)
            update, new_g_acc, new_up_acc = ada_delta_step(grad,
                                                           self.decay,
                                                           self.epsilon,
                                                           grad_acc,
                                                           up_acc)
            # update = tPrint("update for" + param.name)(update)
            return [(param, param - update), (grad_acc, new_g_acc), (up_acc, new_up_acc)]

        def create_gradient_updates(gradients):
            params_to_up = zip(self.params, gradients, self.param_grad_accums,
                               self.param_up_accums)
            return OrderedDict([tup for x in params_to_up for tup in create_update_tuple(*x)])

        self.nn_train_b_soft = theano.function(inputs=[beta, s, y],
                                               outputs=[self.bootstrap_soft_cost],
                                               updates=create_gradient_updates(b_soft_grads),
                                               name="nn_train_one",
                                               allow_input_downcast=True)

        self.nn_train_b_hard = theano.function(inputs=[beta, s, y],
                                               outputs=[self.bootstrap_hard_cost],
                                               updates=create_gradient_updates(b_hard_grads),
                                               name="nn_train_one",
                                               allow_input_downcast=True)

        self.nn_train_sq_err = theano.function(inputs=[s, y],
                                               outputs=[self.squared_error],
                                               updates=create_gradient_updates(sq_err_grads),
                                               name="nn_train_one",
                                               allow_input_downcast=True)

        self.nn_train_cross_ent = theano.function(inputs=[s, y],
                                                  outputs=[self.cross_entropy],
                                                  updates=create_gradient_updates(cross_ent_grads),
                                                  name="nn_train_one",
                                                  allow_input_downcast=True)

        self.nn_train_ind_b_soft = theano.function(inputs=[beta, s, y],
                                                   outputs=[self.individual_b_soft_sum],
                                                   updates=create_gradient_updates(ind_b_soft_grads),
                                                   name="nn_train_one",
                                                   allow_input_downcast=True)

        # self.nn_train_ind_b_hard = theano.function(inputs=[s, y],
        #                                            outputs=[self.individual_b_hard_sum],
        #                                            updates=create_gradient_updates(ind_b_hard_grads),
        #                                            name="nn_train_one",
        #                                            allow_input_downcast=True)

        self.nn_train_ind_cross_ent = theano.function(inputs=[s, y],
                                                      outputs=[self.individual_cross_ent_sum],
                                                      updates=create_gradient_updates(ind_cross_ent_grads),
                                                      name="nn_train_one",
                                                      allow_input_downcast=True)

        self.nn_embed = theano.function([s], embed_test(s), allow_input_downcast=True)
        self.nn_predict = theano.function([s], forward_prop_test(s), allow_input_downcast=True)

    def lookup_vectors(self, sent_idx):
        row_vecs = np.array([self.word_vecs[i] for i in sent_idx])
        return np.transpose(row_vecs)

    def shuffle_training(self):
        self.rand_state.shuffle(self.train_set)

    def reshape_sentences(self, sentences):
        padded = pad_sentence_batch(sentences)
        # input_3d = np.array(map(self.lookup_vectors, padded))
        # shape = input_3d.shape
        # return np.reshape(input_3d, (shape[0], 1, shape[1], shape[2]))
        return padded

    def restructure_batch(self, lst_of_tups):
        sentences, ys = zip(*lst_of_tups)
        return self.reshape_sentences(sentences), ys

    def __train_outline(self, method, epochs, batch_size, stop_crit, shuffle):
        def make_batches(lst, bs):
            return [self.restructure_batch(lst[i:i+bs]) for i in xrange(0, len(lst), bs)]
        not_improving_count = 0
        best_f1 = 0
        for epoch in range(epochs):
            print "Running epoch " + str(epoch)
            sys.stdout.flush()
            tic = time.time()
            batches = make_batches(self.train_set, batch_size)
            for x, y in batches:
                method(x, y)
            if self.valid_set is not None:
                cur_f1 = self.cross_validate(self.valid_set)
                print "F1 score for this epoch:"
                print cur_f1
                if cur_f1 > best_f1:
                    not_improving_count = 0
                    best_f1 = cur_f1
                else:
                    not_improving_count += 1
            print 'epoch %i' % epoch, 'completed in %.2f (sec)' % (time.time()-tic)
            if not_improving_count > stop_crit:
                break
            if shuffle:
                self.shuffle_training()

    def train_b_soft(self, beta=0.95, epochs=5, batch_size=1, stop_crit=5, shuffle=True):
        train = lambda x, y: self.nn_train_b_soft(beta, x, y)
        self.__train_outline(train, epochs, batch_size, stop_crit, shuffle)

    def train_b_hard(self,beta=0.8, epochs=5, stop_crit=5, batch_size=1, shuffle=True):
        train = lambda x, y: self.nn_train_b_hard(beta, x, y)
        self.__train_outline(train, epochs, batch_size, stop_crit, shuffle)

    def train_sq_err(self, epochs=50, batch_size=1, stop_crit=5, shuffle=True):
        self.__train_outline(self.nn_train_sq_err, epochs, batch_size, stop_crit, shuffle)

    def train_cross_ent(self, epochs=50, batch_size=1, stop_crit=5, shuffle=True):
        self.__train_outline(self.nn_train_cross_ent, epochs, batch_size, stop_crit, shuffle)

    def train_multi_b_soft(self, beta=0.95, epochs=50, batch_size=1, stop_crit=5, shuffle=True):
        train = lambda x, y: self.nn_train_ind_b_soft(beta, x, y)
        self.__train_outline(train, epochs, batch_size, stop_crit, shuffle)

    # def train_multi_b_hard(self, beta=0.95, epochs=50, batch_size=1, stop_crit=5, shuffle=True):
    #     train = lambda x, y: self.nn_train_ind_b_hard(beta, x, y)
    #     self.__train_outline(train, epochs, batch_size, stop_crit, shuffle)

    def train_multi_cross_ent(self, epochs=50, batch_size=1, stop_crit=5, shuffle=True):
        self.__train_outline(self.nn_train_ind_cross_ent, epochs, batch_size, stop_crit, shuffle)


    def predict(self, exs, batch_size=1):
        """
        :param exs: lists of tuples(words, word_order, 3d_dep_tensor, leaf_counts)
        :return: matrix of probabilities for each class
        """
        batches = [self.reshape_sentences(exs[i:i + batch_size]) for i in
                   xrange(0, len(exs), batch_size)]
        preds = [self.nn_predict(batch) for batch in batches]
        return [row for mat in preds for row in mat]

    def cross_validate(self, pairs):
        predictions = self.predict([x[0] for x in pairs])
        return sklm.f1_score([np.argmax(x[1]) for x in pairs], map(np.argmax, predictions))

    def embed(self, exs, batch_size=1):
        """
        :param exs: lists of tuples(words, word_order, 3d_dep_tensor, leaf_counts)
        :return: matrix of probabilities for each class
        """
        batches = [self.reshape_sentences(exs[i:i + batch_size]) for i in
                   xrange(0, len(exs), batch_size)]
        preds = [self.nn_embed(batch) for batch in batches]
        return [row for mat in preds for row in mat]

    def save_weights(self, file_name):
        to_save = [x.get_value() for x in self.params]
        pickle.dump(to_save, open(file_name, 'wb'))

    def load_weights(self, file_name):
        """
        This will override any filter sizes you initialized to be those of the saved model
        in the shared variables
        :param file_name:
        :return:
        """
        to_load = pickle.load(open(file_name, 'rb'))
        for s,l in zip(self.params, to_load):
            s.set_value(l)


