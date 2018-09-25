# -*- coding: utf-8 -*-

"""
Implementation of https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
HierarchicalAttention Structure:
1. embedding
2. Word Encoder: word level bi-directional GRU to get rich representation of words
3. Word Attention:word level attention to get important information in a sentence
4. Sentence Encoder: sentence level bi-directional GRU to get rich representation of sentences
5. Sentence Attention: sentence level attention to get important sentence among sentences
6. Fully Connected layer + Softmax
"""

import tensorflow as tf
import tensorflow.contrib as tf_contrib

from loss import errors_mean


class HierarchicalAttention:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 num_sentences, vocab_size, embed_size, hidden_size, is_training, is_classifier,
                 optimizer, need_sentence_level_attention_encoder_flag=True,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0):
        """init all hyperparameters here"""
        # set hyperparamters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.optimizer = optimizer
        self.hidden_size = hidden_size
        self.need_sentence_level_attention_encoder_flag = need_sentence_level_attention_encoder_flag
        self.clip_gradients = clip_gradients
        self.embedded_words = None
        self.hidden_state = None

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.sequence_length = int(self.sequence_length / self.num_sentences)
        if is_classifier:
            # y:[None, ] for classifier
            self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        else:
            # y:[None, num_classes] for regression
            self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.

        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        if is_classifier:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        else:
            # self.accuracy = tf.constant(0.5)
            self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction)

        if not is_training:
            return

        self.loss_val = self.loss()
        self.train_op = self.train()

    def attention_word_level(self, hidden_state):
        """
        input1: self.hidden_state: hidden_state: list,
                len:s entence_length,element: [batch_size*num_sentences,hidden_size*2]
        input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
        return:representation.shape:[batch_size*num_sentences,hidden_size*2]
        """
        # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        hidden_state_ = tf.stack(hidden_state, axis=1)
        # 0) one layer of feed forward network
        # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1, self.hidden_size * 2])
        # hidden_state_:[batch_size*num_sentences*sequence_length,hidden_size*2]
        # W_w_attention_sentence:[,hidden_size*2,,hidden_size*2]
        # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, self.W_w_attention_word) + self.W_b_attention_word)
        # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.sequence_length,
                                                                         self.hidden_size * 2])

        # attention process:
        # 1. get logits for each word in the sentence.
        # 2. get possibility distribution for each word in the sentence.
        # 3. get weighted sum for the sentence as sentence representation.

        # 1) get logits for each word in the sentence.
        # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vector_word)
        # shape:[batch_size*num_sentences,sequence_length]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
        # subtract max for numerical stability (softmax is shift invariant).
        # tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
        # shape:[batch_size*num_sentences,1]
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)

        # 2) get possibility distribution for each word in the sentence.
        # shape:[batch_size*num_sentences,sequence_length]
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)

        # 3) get weighted hidden state by attention vector
        # shape:[batch_size*num_sentences,sequence_length,1]
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        # below sentence_representation'shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        # <----p_attention_expanded:[batch_size*num_sentences,sequence_length,1]
        # hidden_state_:[batch_size*num_sentences,sequence_length,hidden_size*2]
        # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.multiply(p_attention_expanded, hidden_state_)
        # shape:[batch_size*num_sentences,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)
        # shape:[batch_size*num_sentences,hidden_size*2]

        return sentence_representation

    def attention_sentence_level(self, hidden_state_sentence):
        """
        input1: hidden_state_sentence: a list,len:num_sentence,element:[None,hidden_size*4]
        input2: sentence level context vector:[self.hidden_size*2]
        return:representation.shape:[None,hidden_size*4]
        """
        # shape:[None,num_sentence,hidden_size*4]
        hidden_state_ = tf.stack(hidden_state_sentence, axis=1)

        # 0) one layer of feed forward
        # [None*num_sentence,hidden_size*4]
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1, self.hidden_size * 4])
        # shape:[None*num_sentence,hidden_size*2]
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, self.W_w_attention_sentence) +
                                           self.W_b_attention_sentence)
        # [None,num_sentence,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences,
                                                                         self.hidden_size * 2])

        # attention process:
        # 1. get logits for each sentence in the doc.
        # 2. get possibility distribution for each sentence in the doc.
        # 3. get weighted sum for the sentences as doc representation.

        # 1) get logits for each word in the sentence.
        # shape:[None,num_sentence,hidden_size*2]
        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vector_sentence)
        # shape:[None,num_sentence]. that is get logit for each num_sentence.
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
        # subtract max for numerical stability (softmax is shift invariant).
        # tf.reduce_max:computes the maximum of elements across dimensions of a tensor.
        # shape:[None,1]
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)

        # 2) get possibility distribution for each word in the sentence.
        # shape:[None,num_sentence]
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)

        # 3) get weighted hidden state by attention vector(sentence level)
        # shape:[None,num_sentence,1]
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        # shape:[None,num_sentence,hidden_size*2]<---p_attention_expanded:[None,num_sentence,1];
        # hidden_state_:[None,num_sentence,hidden_size*2]
        sentence_representation = tf.multiply(p_attention_expanded, hidden_state_)
        # shape:[None,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)
        # shape:[None,hidden_size*2]

        return sentence_representation

    def inference(self):
        """main computation graph here:
        1. Word Encoder.
        2. Word Attention.
        3. Sentence Encoder
        4. Sentence Attention
        5. linear classifier"""

        # 1.Word Encoder
        # 1.1 embedding of words
        # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
        input_x = tf.split(self.input_x, self.num_sentences, axis=1)
        # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
        input_x = tf.stack(input_x, axis=1)
        # [None,num_sentences,sentence_length,embed_size]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, input_x)
        # [batch_size*num_sentences,sentence_length,embed_size]
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, self.sequence_length, self.embed_size])

        # 1.2 forward gru
        # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        hidden_state_forward_list = self.gru_forward_word_level(embedded_words_reshaped)

        # 1.3 backward gru
        # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        hidden_state_backward_list = self.gru_backward_word_level(embedded_words_reshaped)

        # 1.4 concat forward hidden state and backward hidden state.
        # hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        # hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                             zip(hidden_state_forward_list, hidden_state_backward_list)]

        # 2.Word Attention
        # for each sentence.
        # output:[batch_size*num_sentences,hidden_size*2]
        sentence_representation = self.attention_word_level(self.hidden_state)
        # shape:[batch_size,num_sentences,hidden_size*2]
        sentence_representation = tf.reshape(sentence_representation,
                                             shape=[-1, self.num_sentences, self.hidden_size * 2])
        # with tf.name_scope("dropout"):
        #     shape:[None,hidden_size*4]
        #     sentence_representation = tf.nn.dropout(sentence_representation,keep_prob=self.dropout_keep_prob)

        # 3.Sentence Encoder
        # 3.1) forward gru for sentence
        # a list.length is sentence_length, each element is [None,hidden_size]
        hidden_state_forward_sentences = self.gru_forward_sentence_level(sentence_representation)

        # 3.2) backward gru for sentence
        # a list,length is sentence_length, each element is [None,hidden_size]
        hidden_state_backward_sentences = self.gru_backward_sentence_level(sentence_representation)

        # 3.3) concat forward hidden state and backward hidden state
        # below hidden_state_sentence is a list,len:sentence_length,element:[None,hidden_size*2]
        self.hidden_state_sentence = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward
                                      in zip(hidden_state_forward_sentences, hidden_state_backward_sentences)]

        # 4.Sentence Attention
        # shape:[None,hidden_size*4]
        document_representation = self.attention_sentence_level(self.hidden_state_sentence)
        with tf.name_scope("dropout"):
            # shape:[None,hidden_size*4]
            self.h_drop = tf.nn.dropout(document_representation, keep_prob=self.dropout_keep_prob)

        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            # shape:[None,self.num_classes]==tf.matmul([None,hidden_size*2],[hidden_size*2,self.num_classes])
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
        return logits

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            if self.is_classifier:
                # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
                # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
                # softmax cross entropy loss (sigmoid_cross_entropy_with_logits)
                # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)

                # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
                loss = tf.reduce_mean(losses)
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
        self.learning_rate = learning_rate
        # noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32)))
        # gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer=self.optimizer,
                                                   clip_gradients=self.clip_gradients)
        return train_op

    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        param Xt: Xt:[batch_size*num_sentences,embed_size]
        param h_t_minus_1:[batch_size*num_sentences,embed_size]
        return:
        """
        # update gate: decides how much past information is kept and how much new information is added.
        # z_t:[batch_size*num_sentences,self.hidden_size]
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1, self.U_z) + self.b_z)

        # reset gate: controls how much the past state contributes to the candidate state.
        # r_t:[batch_size*num_sentences,self.hidden_size]
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1, self.U_r) + self.b_r)

        # candiate state h_t~
        # h_t_candiate:[batch_size*num_sentences,self.hidden_size]
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) + r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)

        # new state: a linear combine of pervious hidden state and the current new state h_t~
        # h_t:[batch_size*num_sentences,hidden_size]
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    # Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
    def gru_single_step_sentence_level(self, Xt, h_t_minus_1):
        """
        single step of gru for sentence level
        :param Xt:[batch_size, hidden_size*2]
        :param h_t_minus_1:[batch_size, hidden_size*2]
        :return:h_t:[batch_size,hidden_size]
        """
        # update gate: decides how much past information is kept and how much new information is added.
        # z_t:[batch_size,self.hidden_size]
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_sentence) + tf.matmul(h_t_minus_1,
                                                                         self.U_z_sentence) + self.b_z_sentence)
        # reset gate: controls how much the past state contributes to the candidate state.
        # r_t:[batch_size,self.hidden_size]
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_sentence) + tf.matmul(h_t_minus_1,
                                                                         self.U_r_sentence) + self.b_r_sentence)
        # candiate state h_t~
        # h_t_candiate:[batch_size,self.hidden_size]
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_sentence) + r_t *
                                  (tf.matmul(h_t_minus_1, self.U_h_sentence)) + self.b_h_sentence)

        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    # forward gru for first level: word levels
    def gru_forward_word_level(self, embedded_words):
        """
        param embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        return:forward hidden state: a list.length is sentence_length, each element is
                [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length, axis=1)
        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]

        # demension_1=embedded_words_squeeze[0].get_shape().dims[0]
        # self.hidden_size h_t =int(tf.get_shape(embedded_words_squeeze[0])[0])
        # tf.ones([self.batch_size*self.num_sentences, self.hidden_size])
        # [batch_size*num_sentences,embed_size]
        h_t = tf.ones((self.batch_size * self.num_sentences, self.hidden_size))
        h_t_forward_list = []

        # Xt: [batch_size*num_sentences,embed_size]
        for time_step, Xt in enumerate(embedded_words_squeeze):
            # [batch_size*num_sentences,embed_size]<------Xt:[batch_size*num_sentences,embed_size]
            # h_t:[batch_size*num_sentences,embed_size]
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_forward_list.append(h_t)

        # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        return h_t_forward_list

    # backward gru for first level: word level
    def gru_backward_word_level(self, embedded_words):
        """
        param   embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        return: backward hidden state:a list.length is sentence_length, each element is
                [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length, axis=1)

        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]

        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        embedded_words_squeeze.reverse()
        # demension_1=int(tf.get_shape(embedded_words_squeeze[0])[0])
        # h_t = tf.ones([self.batch_size*self.num_sentences, self.hidden_size])
        h_t = tf.ones((self.batch_size * self.num_sentences, self.hidden_size))
        h_t_backward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_backward_list.append(h_t)
        h_t_backward_list.reverse()

        return h_t_backward_list

    # forward gru for second level: sentence level
    def gru_forward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        # it is a list.length is num_sentences,each element is [batch_size,1,hidden_size*2]
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences, axis=1)
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in
                                           sentence_representation_splitted]

        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0])
        # scalar: batch_size
        h_t = tf.ones((self.batch_size, self.hidden_size * 2))  # TODO
        h_t_forward_list = []

        # Xt:[batch_size, hidden_size*2]
        for time_step, Xt in enumerate(sentence_representation_squeeze):
            # h_t:[batch_size,hidden_size]<---------Xt:[batch_size, hidden_size*2];
            # h_t:[batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level(Xt, h_t)
            h_t_forward_list.append(h_t)

        # a list,length is num_sentences, each element is [batch_size,hidden_size]
        return h_t_forward_list

    # backward gru for second level: sentence level
    def gru_backward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        # it is a list.length is num_sentences,each element is [batch_size,1,hidden_size*2]
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences, axis=1)
        # it is a list.length is num_sentences,each element is [batch_size, hidden_size*2]
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in sentence_representation_splitted]

        sentence_representation_squeeze.reverse()
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0])
        # scalar: batch_size
        h_t = tf.ones((self.batch_size, self.hidden_size * 2))
        h_t_forward_list = []
        # Xt:[batch_size, hidden_size*2]
        for time_step, Xt in enumerate(sentence_representation_squeeze):
            # h_t:[batch_size,hidden_size] <---- Xt:[batch_size, hidden_size*2];
            # h_t:[batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level(Xt, h_t)
            h_t_forward_list.append(h_t)
        h_t_forward_list.reverse()
        # a list,length is num_sentences, each element is [batch_size,hidden_size]
        return h_t_forward_list

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        with tf.name_scope("embedding_projection"):
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            # [embed_size,label_size]
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 4, self.num_classes],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        # GRU parameters:update gate related
        with tf.name_scope("gru_weights_word_level"):
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])

            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        with tf.name_scope("gru_weights_sentence_level"):
            self.W_z_sentence = tf.get_variable("W_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.U_z_sentence = tf.get_variable("U_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.b_z_sentence = tf.get_variable("b_z_sentence", shape=[self.hidden_size * 2])

            # GRU parameters:reset gate related
            self.W_r_sentence = tf.get_variable("W_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.U_r_sentence = tf.get_variable("U_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.b_r_sentence = tf.get_variable("b_r_sentence", shape=[self.hidden_size * 2])

            self.W_h_sentence = tf.get_variable("W_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.U_h_sentence = tf.get_variable("U_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.b_h_sentence = tf.get_variable("b_h_sentence", shape=[self.hidden_size * 2])

        with tf.name_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",
                                                      shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                      initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])

            self.W_w_attention_sentence = tf.get_variable("W_w_attention_sentence",
                                                          shape=[self.hidden_size * 4, self.hidden_size * 2],
                                                          initializer=self.initializer)
            self.W_b_attention_sentence = tf.get_variable("W_b_attention_sentence", shape=[self.hidden_size * 2])
            self.context_vector_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],
                                                       initializer=self.initializer)
            self.context_vector_sentence = tf.get_variable("what_is_the_informative_sentence",
                                                           shape=[self.hidden_size * 2], initializer=self.initializer)
