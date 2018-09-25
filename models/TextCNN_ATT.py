"""
ABCNN Structure
Structure:embedding ---> conv---> max pooling ---> fully connected layer ----> softmax (for classification)
"""
import tensorflow as tf
from loss import errors_mean
import numpy as np


class TextCNN_ATT(object):
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, batchnorm, decay_steps,
                 decay_rate, sequence_length, vocab_size, embed_size, is_training, is_classifier, optimizer,
                 clip_gradients=5.0, decay_rate_big=0.50, initializer=tf.random_normal_initializer(stddev=0.1)):

        # set hyperparamters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.is_classifier = is_classifier
        self.position_embedding_dim = 100
        self.text_embedding_dim = 300
        # ADD learning_rate
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)

        # it is a list of int. e.g. [3,4,5]
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.initializer = initializer
        # Total num of filters
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.clip_gradients = clip_gradients

        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        # add placeholder (X,label) X, y:[None,num_classes]
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_pos1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos2')

        if is_classifier:
            self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        else:
            self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.optimizer = optimizer
        self.batchnorm = batchnorm
        if optimizer == 'Adam':
            self.learning_rate = 0.001

        self.instantiate_weights()
        # [None, self.label_size]. main computation graph is here.
        self.logits = self.inference()
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        # shape:[None,]
        if self.is_classifier:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        else:
            self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction)

    def loss(self, l2_reg_lambda=3.0):
        with tf.name_scope("loss"):
            if self.is_classifier:
                losses = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y, name="cnn_loss")
                loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss
            else:
                loss = errors_mean(self.input_y, self.logits)
        return loss

    def instantiate_weights(self):
        # Embedding layer
        with tf.name_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), name="W_text")
            self.text_embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_x)
            self.text_embedded_chars_expanded = tf.expand_dims(self.text_embedded_chars, -1)

        with tf.name_scope("position-embedding"):
            self.W_position = tf.Variable(tf.random_uniform([pos_vocab_size, self.position_embedding_dim], -1.0, 1.0),
                                          name="W_position")
            self.pos1_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos1)
            self.pos1_embedded_chars_expanded = tf.expand_dims(self.pos1_embedded_chars, -1)
            self.pos2_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos2)
            self.pos2_embedded_chars_expanded = tf.expand_dims(self.pos2_embedded_chars, -1)

        self.embedded_chars_expanded = tf.concat([self.text_embedded_chars_expanded,
                                                  self.pos1_embedded_chars_expanded,
                                                  self.pos2_embedded_chars_expanded], 2)

        self.embed_size = self.text_embedding_dim + 2*self.position_embedding_dim

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)

        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate,
                                                   optimizer=self.optimizer, clip_gradients=self.clip_gradients)

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

    def linear_layer(self, name, x, in_size, out_size, is_regularize=False):
        with tf.variable_scope(name):
            loss_l2 = tf.constant(0, dtype=tf.float32)
            w = tf.get_variable('linear_W', [in_size, out_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('linear_b', [out_size], initializer=tf.constant_initializer(0.1))
            o = tf.nn.xw_plus_b(x, w, b)  # batch_size, out_size
            if is_regularize:
                loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            return o, loss_l2

    def cnn_forward(self, name, sent_pos, lexical, num_filters):
        with tf.variable_scope(name):
            input = tf.expand_dims(sent_pos, axis=-1)
            input_dim = input.shape.as_list()[2]

            # convolutional layer
            pool_outputs = []
            for filter_size in [3, 4, 5]:
                with tf.variable_scope('conv-%s' % filter_size):
                    conv_weight = tf.get_variable('W1', [filter_size, input_dim, 1, num_filters],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
                    conv_bias = tf.get_variable('b1', [num_filters], initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(input, conv_weight, strides=[1, 1, input_dim, 1], padding='SAME')
                    conv = tf.nn.relu(conv + conv_bias)  # batch_size, max_len, 1, num_filters
                    pool = tf.nn.max_pool(conv, ksize=[1, self.sequence_length, 1, 1], strides=[1, max_len, 1, 1],
                                          padding='SAME')  # batch_size, 1, 1, num_filters
                    pool_outputs.append(pool)
            pools = tf.reshape(tf.concat(pool_outputs, 3), [-1, 3 * num_filters])

            # feature
            feature = pools
            if lexical is not None:
                feature = tf.concat([lexical, feature], axis=1)
            return feature

    def create_attention_matrix(self, input1, input2):
        # input1, input2 = [batch, height, width, 1] = [batch, d, s, 1]
        # input2 => [batch, height, 1, width]
        # [batch, width, wdith] = [batch, s, s]
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(input1 - tf.matrix_transpose(input2)), axis=1))
        return 1 / (1 + euclidean)

    def inference(self):
        # main computation graph here TextCNN: 1.embedding-->2.average-->3.linear classifier
        # 1. get emebedding of words in the sentence [None,sentence_length,embed_size]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)

        # 2. loop each filter size. for each filter, do:convolution-pooling layer
        # (a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # a.create filter
                filter_shape = [filter_size, self.embed_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")

                # Apply nonlinearity
                # Add batch norm if param is set.
                if self.batchnorm:
                    conv_bn = self.batch_norm(conv, self.num_filters, tf.cast(self.is_training, tf.bool))
                    # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters].
                    # tf.nn.bias_add:adds `bias` to `value`
                    h = tf.nn.relu(tf.nn.bias_add(conv_bn, b), "relu")
                else:
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")

                # max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                # ksize: A list of ints that has length >= 4. The size of the window for each dim of the input tensor
                # strides: A list of ints that has length >= 4. Stride of the sliding window for each dim of I/P tensor
                # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # 3. combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along
        # one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        # shape should be:[None,num_filters_total]. here this operation has some result as
        # tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        # 4. add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            # [None,num_filters_total]
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            W = tf.get_variable("W", shape=[self.num_filters_total, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
        return self.logits

