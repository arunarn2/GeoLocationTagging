import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from loss import errors_mean

"""
TextRNN Structure : 
    1. embeddding layer, 2.Bi-LSTM layer, 3.concat output, 4.FC layer, 5.softmax
"""


class TextRNN:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size, is_training, batchnorm, is_classifier,
                 optimizer, initializer=tf.random_normal_initializer(stddev=0.1)):
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
        self.batchnorm = batchnorm
        self.num_sampled = 20
        self.is_classifier = is_classifier
        self.optimizer = optimizer

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        if is_classifier:
            self.input_y = tf.placeholder(tf.int32, [None], name="input_y")  # y [None,num_classes]
        else:
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.
        if not is_training:
            return
        self.loss_val = self.loss()  # -->self.loss_nce()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        if is_classifier:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        else:
            self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction, name="Accuracy")
            # self.accuracy = tf.constant(0.5)

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            # [embed_size,label_size]
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.num_classes],
                                                initializer=self.initializer)
            # [label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        # 1.get emebedding of words in the sentence
        # shape:[None,sentence_length,embed_size]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2. Bi-lstm layer
        # define lstm cell:get lstm cell output
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
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words,
                                                     dtype=tf.float32)

        # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        print("outputs:===>", outputs)

        # outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100)
        # dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
        # 3. concat output
        output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
        self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1)

        # [batch_size,hidden_size*2]
        # #output_rnn_last=output_rnn[:,-1,:]
        # [batch_size,hidden_size*2]
        # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        print("output_rnn_last:", self.output_rnn_last)

        # 4. logits(use linear layer)
        with tf.name_scope("output"):
            # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            # [batch_size,num_classes]
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            if self.is_classifier:
                # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
                # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits`
                #         with the softmax cross entropy loss.
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                        logits=self.logits)
                # sigmoid_cross_entropy_with_logits.
                # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
                # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
                loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
                l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                      if 'bias' not in v.name]) * l2_lambda
                loss = loss + l2_losses
            else:
                loss = errors_mean(self.input_y, self.logits)
        return loss

    def loss_nce(self, l2_lambda=0.0001):  # 0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        if self.is_training:  # training
            # labels=tf.reshape(self.input_y,[-1])
            # #[batch_size,1]------>[batch_size,]
            # [batch_size,]----->[batch_size,1]
            labels = tf.expand_dims(self.input_y, 1)
            loss = tf.reduce_mean(
                # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                tf.nn.nce_loss(weights=tf.transpose(self.W_projection),
                               # [hidden_size*2, num_classes]--->[num_classes,hidden_size*2].
                               # nce_weights:A `Tensor` of shape `[num_classes, dim].O.K.
                               biases=self.b_projection,
                               # [label_size]. nce_biases:A `Tensor` of shape `[num_classes]`.
                               labels=labels,
                               # [batch_size,1]. train_labels, # A `Tensor` of type `int64` and
                               # shape `[batch_size,num_true]`. The target classes.
                               inputs=self.output_rnn_last,
                               # [batch_size,hidden_size*2] #A `Tensor` of shape `[batch_size, dim]`.
                               # The forward activations of the input network.
                               num_sampled=self.num_sampled,  # scalar. 100
                               num_classes=self.num_classes, partition_strategy="div"))  # scalar. 1999
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer=self.optimizer)
        return train_op

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


# test phase
def test():
    # below is a function test; if you use this for text classifiction, you need
    # to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes = 49
    learning_rate = 0.01
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100
    is_training = True
    dropout_keep_prob = 1  # 0.5
    textRNN = TextRNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, vocab_size,
                      embed_size, is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size, sequence_length))  # [None, self.sequence_length]
            input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1])
            loss, acc, predict, _ = sess.run(
                [textRNN.loss_val, textRNN.accuracy, textRNN.predictions, textRNN.train_op],
                feed_dict={textRNN.input_x: input_x, textRNN.input_y: input_y,
                           textRNN.dropout_keep_prob: dropout_keep_prob})
            print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)
