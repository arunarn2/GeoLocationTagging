# -*- coding: utf-8 -*-
# Recurrent convolutional neural network for text classification
# TextRNN: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat output, 4.FC layer, 5.softmax
import copy

import tensorflow as tf

from loss import errors_mean


class TextRCNN:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size, is_training, is_classifier, optimizer, batch_norm,
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.activation = tf.nn.relu
        self.is_classifier = is_classifier
        self.batch_norm = batch_norm
        self.optimizer = optimizer

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        if is_classifier:
            self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        else:
            self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        if optimizer == 'Adam':
            self.learning_rate = 0.001

        self.instantiate_weights()
        # [None, self.label_size]. main computation graph is here.
        self.logits = self.inference()
        if not is_training:
            return

        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        if is_classifier:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        else:
            # self.accuracy = tf.constant(0.5)
            self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction)

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
        with tf.name_scope("weights"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)

            self.left_side_first_word = tf.get_variable("left_side_first_word",
                                                        shape=[self.batch_size, self.embed_size],
                                                        initializer=self.initializer)
            self.right_side_last_word = tf.get_variable("right_side_last_word",
                                                        shape=[self.batch_size, self.embed_size],
                                                        initializer=self.initializer)

            self.W_l = tf.get_variable("W_l", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.W_sl = tf.get_variable("W_sl", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.W_sr = tf.get_variable("W_sr", shape=[self.embed_size, self.embed_size], initializer=self.initializer)

            self.b = tf.get_variable("b", [self.embed_size])
            # [embed_size,label_size]
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 3, self.num_classes],
                                                initializer=self.initializer)
            # [label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def get_context_left(self, context_left, embedding_previous):
        """
        :param context_left:
        :param embedding_previous:
        :return: output:[None,embed_size]
        """
        # shape:[batch_size,embed_size]<--context_left:[batch_size,embed_size];W_l:[embed_size,embed_size]
        left_c = tf.matmul(context_left, self.W_l)

        # shape:[batch_size,embed_size]<---embedding_previous;[batch_size,embed_size];W_sl:[embed_size, embed_size]
        left_e = tf.matmul(embedding_previous, self.W_sl)
        # shape:[batch_size,embed_size]
        left_h = left_c + left_e

        # context_left=self.activation(left_h) #shape:[batch_size,embed_size]
        context_left = tf.nn.relu(tf.nn.bias_add(left_h, self.b), "relu")
        # shape:[batch_size,embed_size]
        return context_left

    def get_context_right(self, context_right, embedding_afterward):
        """
        :param context_right:
        :param embedding_afterward:
        :return: output:[None,embed_size]
        """
        # shape:[batch_size,embed_size]<----context_right:[batch_size,embed_size];W_r:[embed_size,embed_size]
        right_c = tf.matmul(context_right, self.W_r)
        # shape:[batch_size,embed_size]<----embedding_afterward:[batch_size,embed_size];W_sr:[embed_size,embed_size]
        right_e = tf.matmul(embedding_afterward, self.W_sr)
        # shape:[batch_size,embed_size]
        right_h = right_c + right_e
        # context_right=self.activation(right_h)
        # #shape:[batch_size,embed_size]
        context_right = tf.nn.relu(tf.nn.bias_add(right_h, self.b), "relu")
        # shape:[batch_size,embed_size]
        return context_right

    def conv_layer_with_recurrent_structure(self):
        """
        input:self.embedded_words:[None,sentence_length,embed_size]
        :return: shape:[None,sentence_length,embed_size*3]
        """
        # 1. get splitted list of word embeddings
        # sentence_length[None,1,embed_size]
        embedded_words_split = tf.split(self.embedded_words, self.sequence_length, axis=1)
        # sentence_length [None,embed_size]
        embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in embedded_words_split]

        # tf.zeros((self.batch_size,self.embed_size))
        embedding_previous = self.left_side_first_word
        # self.left_side_context_first
        context_left_previous = tf.zeros((self.batch_size, self.embed_size))

        # 2. get list of context left
        context_left_list = []
        # sentence_length [None,embed_size]
        for i, current_embedding_word in enumerate(embedded_words_squeezed):
            # [None,embed_size]
            context_left = self.get_context_left(context_left_previous, embedding_previous)
            # append result to list
            context_left_list.append(context_left)
            # assign embedding_previous
            embedding_previous = current_embedding_word
            # assign context_left_previous
            context_left_previous = context_left

        # 3. get context right
        embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()
        embedding_afterward = self.right_side_last_word
        # tf.zeros((self.batch_size,self.embed_size))
        # self.right_side_context_last
        context_right_afterward = tf.zeros((self.batch_size, self.embed_size))
        context_right_list = []
        for j, current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right = self.get_context_right(context_right_afterward, embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward = current_embedding_word
            context_right_afterward = context_right

        # 4.ensemble "left,embedding,right" to output
        output_list = []
        for index, current_embedding_word in enumerate(embedded_words_squeezed):
            # representation's shape:[None,embed_size*3]
            representation = tf.concat([context_left_list[index], current_embedding_word,
                                        context_right_list[index]], axis=1)
            # shape:sentence_length[None,embed_size*3]
            output_list.append(representation)

        # 5. stack list to a tensor
        # shape:[None,sentence_length,embed_size*3]
        output = tf.stack(output_list, axis=1)
        return output

    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.max pooling, 4.FC layer 5.softmax """
        # 1.get emebedding of words in the sentence
        # shape:[None,sentence_length,embed_size]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2. Bi-lstm layer
        # shape:[None,sentence_length,embed_size*3]
        output_conv = self.conv_layer_with_recurrent_structure()
        # 2.1 apply nolinearity
        # b = tf.get_variable("b", [self.embed_size*3])
        # h = tf.nn.relu(tf.nn.bias_add(output_conv, b), "relu")

        # 3. max pooling
        # shape:[None,embed_size*3]
        output_pooling = tf.reduce_max(output_conv, axis=1)
        # 4. logits(use linear layer)
        with tf.name_scope("dropout"):
            # [None,embed_size*3]
            h_drop = tf.nn.dropout(output_pooling, keep_prob=self.dropout_keep_prob)

        # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
        with tf.name_scope("output"):
            # shape:[batch_size,num_classes]<-----h_drop:[None,embed_size*3];
            # b_projection:[hidden_size*3, self.num_classes]
            logits = tf.matmul(h_drop, self.W_projection) + self.b_projection

        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            if self.is_classifier:
                # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
                # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
                #               softmax cross entropy loss.
                # sigmoid_cross_entropy_with_logits.
                # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
                loss = tf.reduce_mean(losses)
                l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                      if 'bias' not in v.name]) * l2_lambda
                loss = loss + l2_losses
            else:
                loss = errors_mean(self.input_y, self.logits)
        return loss

    def batch_norm(self, x, n_out, phase_train):
        """
        Add Batch normalization on convolutional layer.
        Args:
            x:           Tensor
            n_out:       integer
            phase_train: boolean tf.Varialbe, true indicates training phase
        Return:
            bnormed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            bnormed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return bnormed

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer=self.optimizer)
        return train_op
