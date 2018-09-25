# -*- coding: utf-8 -*-
"""
BiLstmTextRelation: check reationship of two questions(Qi,Qj),result(0 or 1). 1 means related,0 means no relation
input_x eg "how much is the computer? EOS price of laptop". 2 different questions are split by a special token: EOS
main graph:1. embeddding layer, 2.Bi-LSTM layer, 3.mean pooling, 4.FC layer, 5.softmax
"""
import tensorflow as tf
from tensorflow.contrib import rnn

from loss import errors_mean


class BiLstmTextRelation:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size, is_training, is_classifier, optimizer,
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
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.optimizer = optimizer

        # add placeholder (X,label)
        # X: input_x e.g. "how much is the computer? EOS price of laptop"
        # X: concat of two sentence, split by EOS.
        # y [None,num_classes]
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        if is_classifier:
            self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        else:
            self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")

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
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        if self.is_classifier:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        else:
            # self.accuracy = tf.constant(0.5)
            self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction)

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

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        with tf.name_scope("embedding"):
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            # [embed_size,label_size], [label_size]
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.num_classes],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def inference(self):
        """
        main computation graph here:
        1. embeddding layer, 2.Bi-LSTM layer,
        3.mean pooling, 4.FC layer, 5.softmax
        """

        # 1.get emebedding of words in the sentence
        # shape:[None,sentence_length,embed_size]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2. Bi-lstm layer
        # define lstm cess:get lstm cell output
        # forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        # backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                                    where:outputs: A tuple (output_fw, output_bw)
        #                                    containing the forward and the backward rnn output `Tensor`.
        # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, dtype=tf.float32)
        # outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>,
        # <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
        print("outputs:===>", outputs)

        # 3. concat output
        # [batch_size,sequence_length,hidden_size*2]
        output_rnn = tf.concat(outputs, axis=2)
        # [batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:]
        # [batch_size,hidden_size*2] #TODO
        output_rnn_pooled = tf.reduce_mean(output_rnn, axis=1)
        # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        print("output_rnn_pooled:", output_rnn_pooled)

        # 4. logits(use linear layer)
        with tf.name_scope("output"):
            # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            # [batch_size,num_classes]
            logits = tf.matmul(output_rnn_pooled, self.W_projection) + self.b_projection
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            if self.is_classifier:
                # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
                # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
                #         softmax cross entropy loss.
                # sigmoid_cross_entropy_with_logits.
                # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
                # print("1. sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
                loss = tf.reduce_mean(losses)
                # print("2. loss.loss:", loss) #shape=()
                l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                      if 'bias' not in v.name]) * l2_lambda
                loss = loss + l2_losses
            else:
                loss = errors_mean(self.input_y, self.logits)
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer=self.optimizer)
        return train_op
