# -*- coding: utf-8 -*-
# seq2seq_attention: 1.word embedding 2.encoder 3.decoder(optional with attention).

import tensorflow as tf
import tensorflow.contrib as tf_contrib

from loss import errors_mean


def extract_argmax_and_embed(embedding, output_projection=None):
    """
    Get a loop_function that extracts the previous symbol and embeds it. Used by decoder.
    param embedding: embedding tensor for symbol
    param output_projection: None or a pair (W, B). If provided, each fed previous output will
    first be multiplied by W and added B.
    return: A loop function
    """

    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
        prev_symbol = tf.argmax(prev, 1)
        emb_prev = tf.gather(embedding, prev_symbol)
        return emb_prev

    return loop_function


def rnn_decoder_with_attention(decoder_inputs, initial_state, cell, loop_function, attention_states, scope=None):
    # 3D Tensor [batch_size x attn_length x attn_size]
    """RNN decoder for the sequence-to-sequence model.
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].it is decoder input.
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].it is the encoded
        vector of input sentences, which represent 'thought vector'
        cell: core_rnn_cell.RNNCell defining the cell function and size.
        loop_function: If not None, this function will be applied to the i-th output
            in order to generate the i+1-st input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].it is represent input X.
        scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
    Returns:
        A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
            shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].
            (Note that in some cases, like basic RNN cell or GRU cell, outputs and
            states can be the same. They are different for LSTM cells though.)
    """
    with tf.variable_scope(scope or "rnn_decoder"):
        print("rnn_decoder_with_attention started...")
        # [batch_size x cell.state_size].
        state = initial_state
        _, hidden_size = state.get_shape().as_list()
        attention_states_original = attention_states
        batch_size, sequence_length, _ = attention_states.get_shape().as_list()
        outputs = []
        prev = None

        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            # 1.get logits of attention for each encoder input.
            # attention_states:[batch_size x attn_length x attn_size]
            # query=state:[batch_size x cell.state_size]
            query = state
            W_a = tf.get_variable("W_a", shape=[hidden_size, hidden_size],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
            # [batch_size,hidden_size]
            query = tf.matmul(query, W_a)
            # [batch_size, 1, hidden_size]
            query = tf.expand_dims(query, axis=1)
            U_a = tf.get_variable("U_a", shape=[hidden_size, hidden_size],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
            U_aa = tf.get_variable("U_aa", shape=[hidden_size])

            # [batch_size*sentence_length,hidden_size]
            attention_states = tf.reshape(attention_states, shape=(-1, hidden_size))
            attention_states = tf.matmul(attention_states, U_a)
            # print("batch_size",batch_size," ;sequence_length:", sequence_length," ; hidden_size:", hidden_size)
            # print("attention_states:", attention_states)
            attention_states = tf.reshape(attention_states, shape=(-1, sequence_length, hidden_size))
            # query_expanded:            [batch_size,1,             hidden_size]
            # attention_states_reshaped: [batch_size,sentence_length,hidden_size]
            # [batch_size,sentence_length,hidden_size]. additive style
            attention_logits = tf.nn.tanh(query + attention_states + U_aa)

            # 2.get possibility of attention
            # batch_size*sequence_length [batch_size*sentence_length,hidden_size]
            attention_logits = tf.reshape(attention_logits, shape=(-1, hidden_size))
            # [hidden_size,1]
            V_a = tf.get_variable("V_a", shape=[hidden_size, 1],
                                  initializer=tf.random_normal_initializer(stddev=0.1))

            # output should be
            # [batch_size*sentence_length,1]<---[batch_size*sentence_length,hidden_size],[hidden_size,1]
            attention_logits = tf.matmul(attention_logits, V_a)
            # attention_logits:[batch_size, sequence_length]
            attention_logits = tf.reshape(attention_logits, shape=(-1, sequence_length))

            # attention_logits = tf.reduce_sum(attention_logits,2)
            # [batch_size x attn_length]
            attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)
            # possibility distribution for each encoder input.it means how much attention or focus for each
            # encoder input. batch_size x attn_length]
            p_attention = tf.nn.softmax(attention_logits - attention_logits_max)

            # 3.get weighted sum of hidden state for each encoder input as attention state
            # [batch_size x attn_length x 1]
            p_attention = tf.expand_dims(p_attention, axis=2)
            # attention_states:[batch_size x attn_length x attn_size];
            # p_attention:[batch_size x attn_length]; [batch_size x attn_length x attn_size]
            attention_final = tf.multiply(attention_states_original, p_attention)
            # [batch_size x attn_size]
            context_vector = tf.reduce_sum(attention_final, axis=1)

            # inp:[batch_size x input_size].it is decoder input;  attention_final:[batch_size x attn_size]
            output, state = cell(inp, state, context_vector)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    print("rnn_decoder_with_attention ended...")
    return outputs, state


class Seq2SeqAttn:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size, hidden_size, is_training, is_classifier, optimizer, decoder_sent_length=6,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0, l2_lambda=0.0001):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.is_classifier = is_classifier
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.decoder_sent_length = num_classes
        self.hidden_size = hidden_size
        self.clip_gradients = clip_gradients
        self.l2_lambda = l2_lambda
        self.optimizer = optimizer

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.decoder_input = tf.placeholder(tf.int32, [None, self.decoder_sent_length], name="decoder_input")
        self.input_y = tf.placeholder(tf.int32, [None, self.decoder_sent_length], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        # logits shape:[batch_size,decoder_sent_length,self.num_classes]
        self.logits = self.inference()

        self.predictions = tf.argmax(self.logits, axis=2, name="predictions")
        # self.accuracy = tf.constant(0.5)
        if self.is_classifier:
            self.accuracy = tf.reduce_mean(tf.cast(self.predictions, tf.float32), name="Accuracy")
        else:
            # self.accuracy = tf.constant(0.5)
            self.accuracy = tf.metrics.mean_squared_error(self.input_y, self.predictions)
        if not is_training:
            return
        self.loss_val = self.loss_seq2seq()
        self.train_op = self.train()

    def inference(self):
        """main computation graph here:
        1. Word embedding.
        2. Encoder with GRU
        3. Decoder using GRU(optional with attention)."""

        # 1.embedding of words
        # [None, self.sequence_length, self.embed_size]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2.encoder with GRU
        # 2.1 forward GRU
        # a list,length is sentence_length, each element is [batch_size,hidden_size]
        hidden_state_forward_list = self.gru_forward(self.embedded_words, self.gru_cell)
        # 2.2 backward GRU
        # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        hidden_state_backward_list = self.gru_forward(self.embedded_words, self.gru_cell, reverse=True)
        # 2.3 concat forward hidden state and backward hidden state. hidden_state:
        # a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        # list,len:sent_len,e:[batch_size,hidden_size*2]
        thought_vector_list = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                               zip(hidden_state_forward_list, hidden_state_backward_list)]

        # 3.Decoder using GRU with attention
        # shape:[batch_size,sentence_length,hidden_size*2]
        thought_vector = tf.stack(thought_vector_list, axis=1)
        # initial_state=tf.reduce_sum(thought_vector, axis=1) #[batch_size,hidden_size*2]
        initial_state = tf.nn.tanh(
            tf.matmul(hidden_state_backward_list[0], self.W_initial_state) + self.b_initial_state)

        # initial_state:[batch_size,hidden_size*2].
        # this is a special cell. because it beside previous hidden state, current input, it also has a
        # context vecotor, which represent attention result.
        cell = self.gru_cell_decoder

        # W_projection:[self.hidden_size * 2, self.num_classes]; b_projection:[self.num_classes]
        output_projection = (self.W_projection, self.b_projection)

        # loop function will be used only at testing, not training.
        loop_function = extract_argmax_and_embed(self.Embedding_label, output_projection) if not \
            self.is_training else None
        # [None, self.sequence_length, self.embed_size]
        attention_states = thought_vector
        # [batch_size,self.decoder_sent_length, embed_size]
        decoder_input_embedded = tf.nn.embedding_lookup(self.Embedding_label, self.decoder_input)
        # it is a list,length is decoder_sent_length, each element is [batch_size,1,embed_size]
        decoder_input_splitted = tf.split(decoder_input_embedded, self.decoder_sent_length, axis=1)
        # it is a list,length is decoder_sent_length, each element is [batch_size,embed_size]
        decoder_input_squeezed = [tf.squeeze(x, axis=1) for x in decoder_input_splitted]

        # rnn_decoder_with_attention(decoder_inputs, initial_state, cell, loop_function,attention_states,scope=None):
        # input1:decoder_inputs:target, shift by one. for example.the target is:"X Y Z",then decoder_inputs
        #                   should be:"START X Y Z" A list of 2D Tensors [batch_size x input_size].
        # input2: initial_state: 2D Tensor with shape  [batch_size x cell.state_size].
        # input3: attention_states:represent X. 3D Tensor [batch_size x attn_length x attn_size].
        # output:?
        # A list.length:decoder_sent_length.each element is:[batch_size x output_size]
        # decoder_output:[batch_size,decoder_sent_length,hidden_size*2]
        # decoder_output:[batch_size*decoder_sent_length,hidden_size*2]

        outputs, final_state = rnn_decoder_with_attention(decoder_input_squeezed, initial_state, cell,
                                                          loop_function, attention_states, scope=None)
        decoder_output = tf.stack(outputs, axis=1)
        decoder_output = tf.reshape(decoder_output, shape=(-1, self.hidden_size * 2))

        with tf.name_scope("dropout"):
            # shape:[None,hidden_size*4]
            decoder_output = tf.nn.dropout(decoder_output, keep_prob=self.dropout_keep_prob)
        # 4. get logits
        with tf.name_scope("output"):
            # logits shape:[batch_size*decoder_sent_length,self.num_classes]==
            # tf.matmul([batch_size*decoder_sent_length,hidden_size*2],[hidden_size*2,self.num_classes])
            # logits shape:[batch_size,decoder_sent_length,self.num_classes]
            logits = tf.matmul(decoder_output, self.W_projection) + self.b_projection
            logits = tf.reshape(logits, shape=(self.batch_size, self.decoder_sent_length, self.num_classes))
        return logits

    def loss_seq2seq(self):
        with tf.name_scope("loss"):
            if self.is_classifier:
                # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
                # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with
                # the softmax cross entropy loss.
                # losses:[batch_size,self.decoder_sent_length]
                # loss_batch:[batch_size]
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
                loss_batch = tf.reduce_sum(losses, axis=1) / self.decoder_sent_length
                loss = tf.reduce_mean(loss_batch)
                l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                      if 'bias' not in v.name]) * self.l2_lambda
                loss = loss + l2_losses
            else:
                loss = errors_mean(self.input_y, self.logits)
            return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate = learning_rate
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer=self.optimizer,
                                                   clip_gradients=self.clip_gradients)
        return train_op

    def gru_cell(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        param Xt: Xt:[batch_size,embed_size]
        param h_t_minus_1:[batch_size,embed_size]
        return: h_t
        """
        # 1.update gate: decides how much past information is kept and how much new information is added.
        # z_t:[batch_size,self.hidden_size]
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1, self.U_z) + self.b_z)

        # 2.reset gate: controls how much the past state contributes to the candidate state.
        # r_t:[batch_size,self.hidden_size]
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1, self.U_r) + self.b_r)

        # candiate state h_t~
        # h_t_candiate:[batch_size,self.hidden_size]
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) + r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)

        # new state: a linear combine of pervious hidden state and the current new state h_t~
        # h_t:[batch_size*num_sentences,hidden_size]
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    def gru_cell_decoder(self, Xt, h_t_minus_1, context_vector):
        """
        single step of gru for word level
        param Xt: Xt:[batch_size,embed_size]
        param h_t_minus_1:[batch_size,embed_size]
        param context_vector. [batch_size,embed_size].this represent the result from attention
                    (weighted sum of input during current decoding step)
        return:
        """
        # 1.update gate: decides how much past information is kept and how much new information is added.
        # z_t:[batch_size,self.hidden_size]
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_decoder) + tf.matmul(h_t_minus_1, self.U_z_decoder) +
                            tf.matmul(context_vector, self.C_z_decoder) + self.b_z_decoder)

        # 2.reset gate: controls how much the past state contributes to the candidate state.
        # r_t:[batch_size,self.hidden_size]
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_decoder) + tf.matmul(h_t_minus_1, self.U_r_decoder) +
                            tf.matmul(context_vector, self.C_r_decoder) + self.b_r_decoder)

        # candiate state h_t~
        # h_t_candiate:[batch_size,self.hidden_size]
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_decoder) + r_t * (tf.matmul(h_t_minus_1, self.U_h_decoder)) +
                                  tf.matmul(context_vector, self.C_h_decoder) + self.b_h_decoder)

        # new state: a linear combine of pervious hidden state and the current new state h_t~
        # h_t:[batch_size*num_sentences,hidden_size]
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t, h_t

    # forward gru for first level: word levels
    def gru_forward(self, embedded_words, gru_cell, reverse=False):
        """
        param embedded_words:[None,sequence_length, self.embed_size]
        return:forward hidden state: a list.length is sentence_length, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        # it is a list,length is sentence_length, each element is [batch_size,1,embed_size]
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length, axis=1)
        # it is a list,length is sentence_length, each element is [batch_size,embed_size]
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]
        h_t = tf.ones((self.batch_size, self.hidden_size))
        h_t_list = []
        if reverse:
            embedded_words_squeeze.reverse()

        # Xt: [batch_size,embed_size]
        for time_step, Xt in enumerate(embedded_words_squeeze):
            # h_t:[batch_size,embed_size]<------Xt:[batch_size,embed_size];h_t:[batch_size,embed_size]
            h_t = gru_cell(Xt, h_t)
            h_t_list.append(h_t)
        if reverse:
            h_t_list.reverse()
        # a list,length is sentence_length, each element is [batch_size,hidden_size]
        return h_t_list

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("decoder_init_state"):
            self.W_initial_state = tf.get_variable("W_initial_state", shape=[self.hidden_size, self.hidden_size * 2],
                                                   initializer=self.initializer)
            self.b_initial_state = tf.get_variable("b_initial_state", shape=[self.hidden_size * 2])

        with tf.name_scope("embedding_projection"):
            # embedding matrix
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            # ,initializer=self.initializer
            self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.num_classes, self.embed_size * 2],
                                                   dtype=tf.float32)
            # [embed_size,label_size]
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.num_classes],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        # GRU parameters:update gate related
        with tf.name_scope("gru_weights_encoder"):
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

        with tf.name_scope("gru_weights_decoder"):
            self.W_z_decoder = tf.get_variable("W_z_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.U_z_decoder = tf.get_variable("U_z_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.C_z_decoder = tf.get_variable("C_z_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.b_z_decoder = tf.get_variable("b_z_decoder", shape=[self.hidden_size * 2])

            # GRU parameters:reset gate related
            self.W_r_decoder = tf.get_variable("W_r_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.U_r_decoder = tf.get_variable("U_r_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.C_r_decoder = tf.get_variable("C_r_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.b_r_decoder = tf.get_variable("b_r_decoder", shape=[self.hidden_size * 2])

            self.W_h_decoder = tf.get_variable("W_h_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.U_h_decoder = tf.get_variable("U_h_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.C_h_decoder = tf.get_variable("C_h_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.b_h_decoder = tf.get_variable("b_h_decoder", shape=[self.hidden_size * 2])

        with tf.name_scope("full_connected"):
            self.W_fc = tf.get_variable("W_fc", shape=[self.hidden_size * 2, self.hidden_size])
            self.a_fc = tf.get_variable("a_fc", shape=[self.hidden_size])
