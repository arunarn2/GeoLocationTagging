"""
TextCNN Structure
Structure:embedding ---> conv---> max pooling ---> fully connected layer ----> softmax (for classification)
"""
import tensorflow as tf

from loss import errors_mean


class TextCNN(object):
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, batchnorm, decay_steps,
                 decay_rate, sequence_length, vocab_size, embed_size, is_training, is_classifier, optimizer,
                 clip_gradients=5.0, decay_rate_big=0.50, initializer=tf.truncated_normal_initializer(stddev=0.1)):

        # set hyperparamters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.is_classifier = is_classifier
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

        # add placeholder (X,label) X, y:[None,num_classes]
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
        # shape:[None,]
        if self.is_classifier:
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        else:
            self.accuracy = tf.constant(0.5)

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            if self.is_classifier:
                # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
                # output: A 1-D `Tensor` of length `batch_size` of the same type as
                # `logits` with the softmax cross entropy loss. sigmoid_cross_entropy_with_logits.
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)

                loss = tf.reduce_mean(losses)
                l2_losses = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
                loss = loss + l2_losses
            else:
                loss = errors_mean(self.input_y, self.logits)
        return loss

    def instantiate_weights(self):
        # define all weights here
        # embedding matrix
        with tf.name_scope("embedding"):
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            # embed_size,label_size]
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer)
            # [label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)

        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate,
                                                   optimizer=self.optimizer, clip_gradients=self.clip_gradients)

        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # train_op = optimizer.apply_gradients(zip(gradients, variables))

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
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1,
                                                                     self.num_filters],
                                         initializer=self.initializer)

                # b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a
                # filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                #       A 4-D tensor. The dim order is determined by the value of `data_format`,see below for details.
                # 1)each filter with conv2d's output a shape:
                #       [1,sequence_length-filter_size+1,1,1];
                # 2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];
                # 3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # input data format:NHWC:[batch, height, width, channels];output:4-D
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")

                # c. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])

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
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
        return logits
