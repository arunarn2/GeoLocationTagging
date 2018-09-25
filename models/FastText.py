import tensorflow as tf

from loss import errors_mean

"""
fast text. using: very simple model;n-gram to captrue location information;
h-softmax to speed up training/inference
"""


class FastText:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 num_sampled, sequence_length, vocab_size, embed_size, is_training,
                 batchnorm, is_classifier, optimizer):
        # init all hyperparameters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.sentence_len = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.batchnorm = batchnorm
        self.is_classifier = is_classifier
        self.optimizer = optimizer

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_len], name="input_x")
        if is_classifier:
            self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        else:
            self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.instantiate_weights()
        self.logits = self.inference()
        # [None, self.label_size]
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        # shape:[None,]
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        if is_classifier:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        else:
            # self.accuracy = tf.constant(0.5)
            self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction)

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size])
        self.W = tf.get_variable("W", [self.embed_size, self.num_classes])
        self.b = tf.get_variable("b", [self.num_classes])

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.get emebedding of words in the sentence
        # [None,self.sentence_len,self.embed_size]
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2.average vectors, to get representation of the sentence
        # [None,self.embed_size]
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)

        # 3.linear classifier layer
        # [None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b
        return logits

    def loss(self, l2_lambda=0.01):  # 0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        if self.is_classifier:
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            if self.is_training:  # training
                # [batch_size,1]------>[batch_size,]
                labels = tf.reshape(self.input_y, [-1])
                # [batch_size,]----->[batch_size,1]
                labels = tf.expand_dims(labels, 1)

                loss = tf.reduce_mean(
                    # inputs: A `Tensor` of shape `[batch_size, dim]`.
                    # The forward activations of the input network.
                    tf.nn.nce_loss(weights=tf.transpose(self.W),
                                   # [embed_size, num_classes]--->[num_classes,embed_size]. nce_weights:
                                   # A `Tensor` of shape `[num_classes, dim].O.K.
                                   # [num_classes]. nce_biases:A `Tensor` of shape `[num_classes]`.
                                   biases=self.b,
                                   labels=labels,
                                   # [batch_size,1]. train_labels, # A `Tensor` of type `int64` and
                                   # shape `[batch_size,num_true]`. The target classes.
                                   inputs=self.sentence_embeddings,
                                   # [None,self.embed_size] #A `Tensor` of shape `[batch_size, dim]`.
                                   # The forward activations of the input network.
                                   num_sampled=self.num_sampled,  # scalar. 100
                                   num_classes=self.num_classes, partition_strategy="div"))  # scalar. 49 or 2
            else:  # eval/inference
                # logits = tf.matmul(self.sentence_embeddings, tf.transpose(self.W))
                # matmul([None,self.embed_size])--->
                # logits = tf.nn.bias_add(logits, self.b)
                labels_one_hot = tf.one_hot(self.input_y, self.num_classes)

                # [batch_size]---->[batch_size,num_classes]
                # sigmoid_cross_entropy_with_logits: Computes sigmoid cross entropy given `logits`.
                # Measures the probability error in discrete classification tasks in which each class
                # is independent and not mutually exclusive.  For instance, one could perform multilabel
                # classification where a picture can contain both an elephant and a dog at the same time.
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)

                # labels:[batch_size,num_classes];logits:[batch, num_classes]
                print("loss0:", loss)
                loss = tf.reduce_sum(loss, axis=1)
                print("loss1:", loss)  # shape=(?,)

            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                  if 'bias' not in v.name]) * l2_lambda
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
